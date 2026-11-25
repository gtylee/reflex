"""Opt-in storage/ledger scaling benchmark; no model calls are made."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import statistics
import platform
import sys
import tempfile
from time import perf_counter
import tracemalloc
from typing import Iterable

import duckdb
from reflex.core.models.events import Event, stable_id
from reflex.core.storage import EventStore


def benchmark_event_store(sizes: Iterable[int] = (1_000, 10_000, 100_000), *,
                          batch_size: int = 2_000, output_dir: str | Path | None = None) -> dict:
    """Measure Python event construction + batched DuckDB insert/query/replay.

    Results are machine-specific and include environment metadata; this is not
    an apples-to-apples SQL/vectorized/agent comparison.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    records = []
    base = datetime(2025, 1, 1, tzinfo=timezone.utc)
    with tempfile.TemporaryDirectory(prefix="reflex-perf-") as temp_dir:
        root = Path(output_dir) if output_dir else Path(temp_dir)
        root.mkdir(parents=True, exist_ok=True)
        for size in sizes:
            if size < 1:
                raise ValueError("event sizes must be positive")
            db_path = Path(temp_dir) / f"events-{size}.duckdb"
            store = EventStore(db_path)
            latencies: list[float] = []
            tracemalloc.start()
            inserted = 0
            started = perf_counter()
            for first in range(0, size, batch_size):
                batch = [Event(
                    event_id=stable_id("performance", str(size), str(index)),
                    event_type="performance.synthetic", event_time=base + timedelta(seconds=index),
                    entity_ids=(f"entity-{index % 128}",),
                    attributes={"value": (index % 997) / 997, "group": index % 8},
                    source="performance-generator-v1", provenance={"sequence": index},
                ) for index in range(first, min(first + batch_size, size))]
                batch_start = perf_counter()
                inserted += store.add_events(batch)
                latencies.append(perf_counter() - batch_start)
            write_seconds = perf_counter() - started
            current_bytes, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            query_start = perf_counter()
            sample = store.events(limit=min(size, 100))
            query_seconds = perf_counter() - query_start
            replay_start = perf_counter()
            replayed = store.add_events(sample)
            replay_seconds = perf_counter() - replay_start
            event_count_after_replay = store.count_events()
            store.close()
            db_bytes = sum(path.stat().st_size for path in
                           (db_path, Path(str(db_path) + ".wal")) if path.exists())
            records.append({
                "events": size, "inserted": inserted, "event_count_after_replay": event_count_after_replay,
                "write_seconds": round(write_seconds, 6),
                "events_per_second": round(size / write_seconds, 3) if write_seconds else None,
                "batch_latency_p50_seconds": round(statistics.median(latencies), 6),
                "batch_latency_p95_seconds": round(_percentile(latencies, 0.95), 6),
                "sample_query_seconds": round(query_seconds, 6), "sample_events": len(sample),
                "idempotent_replay_inserted": replayed, "replay_seconds": round(replay_seconds, 6),
                "peak_python_bytes": peak_bytes, "current_python_bytes": current_bytes,
                "duckdb_file_bytes": db_bytes, "batch_size": batch_size,
            })
    result = {"benchmark": "reflex-event-store-v1", "machine_specific": True,
              "environment": {"platform": platform.platform(), "python": sys.version.split()[0],
                              "duckdb": duckdb.__version__},
              "model_calls": 0, "measurements": records,
              "limitations": ["Measures Python generation and DuckDB append together.",
                              "No SQL-only, vectorized, stateful, or agent baseline is included.",
                              "Results vary by hardware, runtime, filesystem, and DuckDB version."]}
    if output_dir:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "performance.json").write_text(json.dumps(result, indent=2) + "\n")
        (destination / "performance.svg").write_text(_throughput_svg(records))
    return result


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * quantile))]


def _throughput_svg(records: list[dict]) -> str:
    width, height, margin = 720, 400, 60
    max_value = max((item["events_per_second"] or 0 for item in records), default=1) or 1
    bar_width = max(28, min(90, (width - 2 * margin) // max(1, len(records) * 2)))
    slot = (width - 2 * margin) / max(1, len(records))
    bars = []
    for index, item in enumerate(records):
        value = item["events_per_second"] or 0
        bar_height = (height - 2 * margin) * value / max_value
        x = margin + index * slot + (slot - bar_width) / 2
        y = height - margin - bar_height
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="#5279c7"/>')
        bars.append(f'<text x="{x + bar_width / 2:.1f}" y="{height - margin + 22}" text-anchor="middle" font-size="12">{item["events"]:,}</text>')
        bars.append(f'<text x="{x + bar_width / 2:.1f}" y="{max(18, y - 8):.1f}" text-anchor="middle" font-size="11">{value:.0f}/s</text>')
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
            '<rect width="100%" height="100%" fill="white"/>'
            '<text x="360" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">REFLEX Event Store Throughput</text>'
            f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#555"/>'
            + "".join(bars) +
            f'<text x="360" y="{height-12}" text-anchor="middle" font-family="sans-serif" font-size="12">events per run</text></svg>')


__all__ = ["benchmark_event_store"]
