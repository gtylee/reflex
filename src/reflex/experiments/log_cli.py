"""CLI adapter for benchmark-format event-log files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reflex.core.storage import EventStore
from .log_adapters import read_bpi2017, read_loghub


def build_parser() -> argparse.ArgumentParser:
    """Build commands for the supported experiment dataset file formats."""
    parser = argparse.ArgumentParser(prog="reflex experiments ingest")
    sub = parser.add_subparsers(dest="format", required=True)
    bpi = sub.add_parser("bpi")
    bpi.add_argument("path")
    bpi.add_argument("--db", default="events.duckdb")
    loghub = sub.add_parser("loghub")
    loghub.add_argument("path")
    loghub.add_argument("--db", default="events.duckdb")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Ingest one local benchmark file into the generic EventStore."""
    args = build_parser().parse_args(argv)
    if args.format == "bpi":
        events = read_bpi2017(args.path)
    else:
        events = read_loghub(args.path)
    store = EventStore(args.db)
    try:
        inserted = store.add_events(events)
        total = store.count_native_events()
    finally:
        store.close()
    print(json.dumps({"inserted": inserted, "total": total,
                      "db": str(Path(args.db).resolve())}, indent=2))
    return 0


__all__ = ["build_parser", "main"]
