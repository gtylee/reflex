"""Repeatable, opt-in replay of a pinned public GDELT day."""

from __future__ import annotations

import json
from hashlib import sha256
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from reflex.core.storage import EventStore
from .datasets import download_gdelt_day


def replay_gdelt_day(date: str, output_dir: str | Path, *, limit: int = 10_000,
                     batch_size: int = 2_000) -> dict:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    events, source = download_gdelt_day(date, limit=limit)
    if not events:
        raise ValueError("the selected GDELT archive contained no events")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    event_path = root / "events.jsonl"
    event_path.write_text("".join(json.dumps(event.model_dump(mode="json"), sort_keys=True) + "\n"
                                    for event in events))
    event_stream_sha256 = sha256(event_path.read_bytes()).hexdigest()
    database_path = root / "replay.duckdb"
    if database_path.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        database_path = root / f"replay-{date}-{source['archive_sha256'][:8]}-{stamp}.duckdb"
    store = EventStore(database_path)
    started = perf_counter()
    inserted = 0
    for offset in range(0, len(events), batch_size):
        inserted += store.add_events(events[offset:offset + batch_size])
    write_seconds = perf_counter() - started
    replayed_events = store.events()
    chronological = all(left.event_time <= right.event_time
                        for left, right in zip(replayed_events, replayed_events[1:]))
    duplicate_replay_inserted = store.add_events(replayed_events[:min(1000, len(replayed_events))])
    count = store.count_events()
    store.close()
    database_bytes = sum(path.stat().st_size for path in
                         (database_path, Path(str(database_path) + ".wal")) if path.exists())
    report = {
        "experiment": "gdelt-public-event-replay-v1", **source,
        "replay_event_count": count, "inserted_event_count": inserted,
        "chronological_after_replay": chronological,
        "event_stream_sha256": event_stream_sha256,
        "duplicate_replay_inserted": duplicate_replay_inserted,
        "batch_size": batch_size, "write_seconds": write_seconds,
        "events_per_second": count / write_seconds if write_seconds else None,
        "database_bytes": database_bytes, "event_file": str(event_path),
        "database_file": str(database_path),
        "report_file": str(root / "replay-report.json"),
        "limitations": ["GDELT Events rows contain structured event metadata, not complete article text.",
                        "This is an ingestion/replay measurement, not a relevance or market-prediction score."],
    }
    (root / "replay-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


__all__ = ["replay_gdelt_day"]
