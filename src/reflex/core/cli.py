"""CLI for reusable REFLEX event, storage, SQL Lens, and state operations.

Synthetic datasets, domain feature profiles, and benchmark commands live in
``reflex.experiments``. This module contains only commands that operate on an
explicit local event store or user-supplied input file/query.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .lenses import SqlLens
from .models.lenses import CandidateStatus, LensCandidate
from .policy import BoundedTools, PolicyEngine
from .storage import EventStore, StateStore


def _dt(value: str):
    """Parse an ISO-8601 timestamp for event-time bounds."""
    from datetime import datetime

    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _read_sql(args: argparse.Namespace) -> str:
    """Load the query definition supplied for a generic SQL Lens replay."""
    if args.sql and args.sql_file:
        raise ValueError("provide --sql or --sql-file, not both")
    if args.sql_file:
        return Path(args.sql_file).read_text(encoding="utf-8")
    if args.sql:
        return args.sql
    raise ValueError("generic replay requires --sql or --sql-file; use an experiment profile for sample lenses")


def build_parser() -> argparse.ArgumentParser:
    """Build the parser for reusable local operations, without demo profiles."""
    parser = argparse.ArgumentParser(prog="reflex", description="REFLEX reusable event and state operations")
    sub = parser.add_subparsers(dest="command", required=True)

    replay = sub.add_parser("replay", help="run a user-supplied SQL Lens over event-time batches")
    replay.add_argument("--db", default="events.duckdb")
    replay.add_argument("--state", default="state.sqlite")
    query = replay.add_mutually_exclusive_group(required=True)
    query.add_argument("--sql", help="one SELECT/WITH query")
    query.add_argument("--sql-file", help="UTF-8 file containing one SELECT/WITH query")
    replay.add_argument("--lens-id", default="sql.user")
    replay.add_argument("--version", default="1.0")
    replay.add_argument("--description", default="User-supplied deterministic SQL Lens")
    replay.add_argument("--batch-days", type=int, default=5)
    replay.add_argument("--start")
    replay.add_argument("--end")
    replay.add_argument("--limit", type=int, default=1000)

    export = sub.add_parser("export-parquet", help="export a time-ordered Parquet event snapshot")
    export.add_argument("--db", default="events.duckdb")
    export.add_argument("--out", required=True)

    state_cmd = sub.add_parser("state", help="inspect durable state and its transition audit")
    state_cmd.add_argument("--db", default="state.sqlite")
    state_cmd.add_argument("--namespace", choices=sorted(StateStore.NAMESPACES))
    state_cmd.add_argument("--audit", action="store_true")

    lens = sub.add_parser("lens", help="manage SQL Lens candidates")
    lens.add_argument("--state", default="state.sqlite")
    lens_sub = lens.add_subparsers(dest="lens_command", required=True)
    propose = lens_sub.add_parser("propose")
    propose.add_argument("spec")
    for command in ("approve", "activate", "retire"):
        item = lens_sub.add_parser(command)
        item.add_argument("candidate_id")
    validate = lens_sub.add_parser("validate")
    validate.add_argument("candidate_id")
    validate.add_argument("--db", default="events.duckdb")
    validate.add_argument("--start", required=True)
    validate.add_argument("--end", required=True)
    validate.add_argument("--run-id", default="candidate-validation")
    return parser


def _replay(args: argparse.Namespace) -> int:
    sql = _read_sql(args)
    start, end = _dt(args.start) if args.start else None, _dt(args.end) if args.end else None
    if (start is None) != (end is None):
        raise ValueError("replay requires both --start and --end, or neither")
    if args.batch_days <= 0:
        raise ValueError("batch-days must be positive")
    store = EventStore(args.db)
    state = StateStore(args.state)
    lens = SqlLens(args.lens_id, args.version, args.description, sql)

    def state_at_row(row: dict[str, Any]) -> dict[str, Any]:
        """Attach only state visible no later than the selected event."""
        snapshot = state.snapshot_at("operational", row["event_time"])
        snapshot["as_of"] = row["event_time"].isoformat()
        return snapshot

    results = []
    try:
        if start and end:
            batches = []
            cursor = start
            from datetime import timedelta

            while cursor < end:
                batch_end = min(cursor + timedelta(days=args.batch_days), end)
                batches.append((cursor, batch_end, 0))
                cursor = batch_end
        else:
            batches = store.batches(args.batch_days)

        for batch_start, batch_end, _count in batches:
            observations = lens.replay(store, start=batch_start, end=batch_end,
                                      state_snapshot=state_at_row, limit=args.limit)
            store.save_observations(observations)
            results.append({"start": batch_start.isoformat(), "end": batch_end.isoformat(),
                            "observations": len(observations)})
    finally:
        state.close()
        store.close()

    print(json.dumps({"lens_id": lens.definition.lens_id, "version": lens.definition.version,
                      "batches": results, "observations_total": sum(row["observations"] for row in results)}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run a reusable event-store command and return its process exit code."""
    args = build_parser().parse_args(argv)
    if args.command == "replay":
        return _replay(args)
    if args.command == "export-parquet":
        store = EventStore(args.db)
        try:
            count = store.export_parquet(args.out)
        finally:
            store.close()
        print(json.dumps({"rows": count, "path": str(Path(args.out).resolve())}, indent=2))
        return 0
    if args.command == "state":
        state = StateStore(args.db)
        try:
            data = state.audit(args.namespace) if args.audit else state.snapshot(args.namespace)
        finally:
            state.close()
        print(json.dumps(data, indent=2, default=str))
        return 0
    return _run_lens_command(args)


def _run_lens_command(args: argparse.Namespace) -> int:
    """Handle candidate lifecycle operations against the local state store."""
    state = StateStore(args.state)
    event_store = None
    try:
        if args.lens_command == "propose":
            candidate = LensCandidate.model_validate(json.loads(Path(args.spec).read_text()))
            state.put_candidate(candidate)
            result: Any = candidate.model_dump(mode="json")
        elif args.lens_command == "validate":
            event_store = EventStore(args.db)
            policy = PolicyEngine(state, run_id=args.run_id, max_investigations=0, max_queries=5)
            tools = BoundedTools(event_store, state, run_id=args.run_id)
            result = tools.validate_lens(args.candidate_id, event_store,
                                         validation_start=_dt(args.start), validation_end=_dt(args.end))
        else:
            transitions = {"approve": CandidateStatus.APPROVED.value,
                           "activate": CandidateStatus.ACTIVE.value,
                           "retire": CandidateStatus.RETIRED.value}
            result = state.transition_candidate(args.candidate_id,
                                                transitions[args.lens_command]).model_dump(mode="json")
        print(json.dumps(result, indent=2, default=str))
        return 0
    finally:
        if event_store is not None:
            event_store.close()
        state.close()


__all__ = ["build_parser", "main"]
