"""Build a training-only taxi lens package and inspect its held-out results."""

import argparse
import csv
import hashlib
import json
from datetime import UTC, datetime, timedelta
from itertools import groupby
from pathlib import Path
from tempfile import TemporaryDirectory

from reflex.core import build_lens_package, load_lens_package
from reflex.core.models.events import NativeEvent
from reflex.core.storage import EventStore, StateStore
from reflex.registry import FunctionRegistry
from reflex.runtime import ReactionRuntime
from lens import CANDIDATES, EVENT, bind, feature_row

# The order of these blocks is fixed before scoring the held-out period.
CHECK = datetime(2014, 9, 1, tzinfo=UTC)
CAL = datetime(2014, 10, 1, tzinfo=UTC)
EVAL = datetime(2014, 11, 1, tzinfo=UTC)
STEP = timedelta(minutes=30)


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_series(path):
    with path.open(newline="") as stream:
        rows = [(datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC),
                 int(row["value"])) for row in csv.DictReader(stream)]
    rows.sort()
    if any(value < 0 or at.minute not in (0, 30) or at.second for at, value in rows):
        raise ValueError("expected nonnegative counts on half-hour boundaries")
    if any(a[0] == b[0] for a, b in zip(rows, rows[1:])):
        raise ValueError("duplicate half-hour timestamp")
    return rows


def _row(at, count):
    source_id = f"nyc_taxi/{at:%Y-%m-%d %H:%M}"
    return feature_row(at, float(count), source_id)


def flag_cases(flags):
    """Combine consecutive flagged half-hours and preserve each source ID."""
    cases = []
    for _, group in groupby(enumerate(flags), key=lambda item: item[1]["time"] - item[0] * STEP):
        points = [point for _, point in group]
        peak = max(points, key=lambda point: point["score"])
        cases.append({"start": points[0]["time"].isoformat(), "last": points[-1]["time"].isoformat(),
                      "points": len(points), "peak_id": peak["source_id"], "peak_score": peak["score"],
                      "evidence": [point["source_id"] for point in points]})
    return cases


def replay(package_path, point):
    package = load_lens_package(package_path, CANDIDATES)
    registry = FunctionRegistry()
    graph = bind(package, registry)
    event = NativeEvent(event_id=point["source_id"], event_type=EVENT, event_time=point["time"],
                        entity_ids=("nyc-taxi",), attributes={"passengers": point["passengers"]})
    with TemporaryDirectory() as scratch:
        store = EventStore(Path(scratch) / "events.duckdb")
        state = StateStore(Path(scratch) / "state.sqlite")
        try:
            run = ReactionRuntime(store, state, registry).process(event, graph)
            if run.status != "succeeded" or len(run.emitted_event_ids) != 1:
                raise RuntimeError(f"Replay failed: {run.errors}")
            observed = next(e for e in store.events() if e.event_id == run.emitted_event_ids[0])
            if observed.payload["value"]["score"] != point["score"]:
                raise ValueError("saved lens and offline score disagree")
            return observed.model_dump(mode="json")
        finally:
            state.close()
            store.close()


def build(source, out):
    source_rows = read_series(source)
    rows = [(_row(at, count), at, count) for at, count in source_rows]
    fit_rows = [row for row, at, _ in rows if at < CHECK]
    selection_rows = [row for row, at, _ in rows if CHECK <= at < CAL]
    calibration_rows = [row for row, at, _ in rows if CAL <= at < EVAL]
    later = [(row, at, count) for row, at, count in rows if at >= EVAL]
    if not later:
        raise ValueError("need later half-hours to evaluate this pilot")
    package = build_lens_package(
        fit_rows=fit_rows, selection_rows=selection_rows, calibration_rows=calibration_rows,
        candidates=CANDIDATES, goal="Review unexpectedly quiet half-hours across NYC taxi demand",
        calibration_rate=0.01, source_fingerprint=hashlib.sha256(source.read_bytes()).hexdigest())
    out.mkdir(parents=True, exist_ok=True)
    package.save(out / "lens.json")
    observations = package.score([row for row, _, _ in later])
    points = []
    for (row, at, _), observation in zip(later, observations, strict=True):
        value = dict(observation.value)
        points.append(dict(value, time=at, source_id=row.supporting_event_ids[0],
                           flagged=value["is_anomaly"]))
    with (out / "scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=points[0].keys())
        writer.writeheader()
        writer.writerows(points)
    flags = [point for point in points if point["flagged"]]
    cases = flag_cases(flags)
    save(out / "cases.json", cases)
    peak = max(points, key=lambda p: p["score"])
    same_peer = [p for p in points if not p["flagged"] and p["peer"] == peak["peer"]]
    typical = min(same_peer, key=lambda p: abs(p["passengers"] - p["expected"])) if same_peer else None
    save(out / "replay.json", replay(out / "lens.json", peak))
    return {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "candidate_ref": package.manifest["candidate_ref"], "lens_version": package.version,
            "training_selection_loss": package.manifest["selection_loss"],
            "calibration_threshold": package.threshold,
            "calibration_flags": package.manifest["calibration_alert_count"],
            "evaluation_rows": len(points), "flagged_half_hours": len(flags), "review_cases": len(cases),
            "missing_half_hours": sum((b[0] - a[0]) // STEP - 1 for a, b in zip(source_rows, source_rows[1:])),
            "peak": peak["source_id"], "unflagged_peer": typical["source_id"] if typical else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    report = build(args.source, args.out)
    save(args.out / "summary.json", report)
    print(json.dumps(report, indent=2))
