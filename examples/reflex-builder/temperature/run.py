"""Freeze a lens package and apply the separate weekly review budget."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from reflex.core import build_lens_package, load_lens_package
from reflex.core.models.events import NativeEvent
from reflex.core.storage import EventStore, StateStore
from reflex.registry import FunctionRegistry
from reflex.runtime import ReactionRuntime
from lens import CANDIDATES, EVENT_TYPE, bind, feature_row

CHECK = datetime(2013, 10, 1, tzinfo=UTC)
CAL = datetime(2013, 11, 1, tzinfo=UTC)
EVAL = datetime(2014, 1, 1, tzinfo=UTC)
HOUR = timedelta(hours=1)
LIMIT = 3
SOURCE_KEY = "realKnownCause/ambient_temperature_system_failure.csv"


def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_rows(path):
    with path.open(newline="") as stream:
        rows = [(datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC),
                 float(row["value"])) for row in csv.DictReader(stream)]
    if any(not math.isfinite(v) or at.minute or at.second for at, v in rows):
        raise ValueError("expected finite hourly measurements")
    if any(right[0] <= left[0] for left, right in zip(rows, rows[1:])):
        raise ValueError("expected timestamps in increasing order without duplicates")
    return rows


def week(at):
    return (at.date() - timedelta(days=at.weekday())).isoformat()


def normalized(rows):
    return [(feature_row(at, value, f"office/{at:%Y-%m-%dT%H}"), at, value)
            for at, value in rows]


def scored(rows, package):
    observations = package.score([row for row, _, _ in rows])
    return [dict(observation.value, time=at, source_id=observation.supporting_event_ids[0])
            for observation, (_, at, _) in zip(observations, rows, strict=True)]


def review(points, cutoff):
    """Apply this example's first-crossing decision and weekly queue cap."""
    used = Counter()
    cases = []
    prev = None
    for p in points:
        if p["score"] <= cutoff:
            prev = None
            continue
        new_case = prev is None or p["time"] - prev["time"] != HOUR or p["direction"] != prev["direction"]
        if new_case:
            w = week(p["time"])
            admitted = used[w] < LIMIT
            used[w] += int(admitted)
            cases.append({"opened": p["time"].isoformat(), "week_start": w, "queued": admitted,
                          "opening_score": p["score"], "observed": p["observed"], "expected": p["expected"],
                          "direction": p["direction"], "evidence": []})
        cases[-1]["evidence"].append(p["source_id"])
        prev = p
    return cases


def replay(package_path, point):
    package = load_lens_package(package_path, CANDIDATES)
    registry = FunctionRegistry()
    graph = bind(package, registry)
    event = NativeEvent(event_id=point["source_id"], event_type=EVENT_TYPE, event_time=point["time"],
                        entity_ids=("office-sensor",), attributes={"temperature": point["observed"]})
    with TemporaryDirectory() as tmp:
        store = EventStore(Path(tmp) / "events.duckdb")
        state = StateStore(Path(tmp) / "state.sqlite")
        try:
            run = ReactionRuntime(store, state, registry).process(event, graph)
            if run.status != "succeeded" or len(run.emitted_event_ids) != 1:
                raise RuntimeError(f"Unexpected runtime outcome: {run.errors}")
            obs = next(o for o in store.events() if o.event_id == run.emitted_event_ids[0])
            if obs.payload["value"]["score"] != point["score"] or obs.parent_event_ids != (event.event_id,):
                raise RuntimeError("Saved lens did not reproduce the source observation")
            return obs.model_dump(mode="json")
        finally:
            state.close()
            store.close()


def build(source, out):
    source_rows = read_rows(source)
    rows = normalized(source_rows)
    fit_rows = [row for row, at, _ in rows if at < CHECK]
    selection_rows = [row for row, at, _ in rows if CHECK <= at < CAL]
    calibration_rows = [row for row, at, _ in rows if CAL <= at < EVAL]
    later = [(row, at, value) for row, at, value in rows if at >= EVAL]
    if not later:
        raise ValueError("need later readings to evaluate this pilot")
    package = build_lens_package(
        fit_rows=fit_rows, selection_rows=selection_rows, calibration_rows=calibration_rows,
        candidates=CANDIDATES, goal="Review unusual high or low office temperatures",
        # The old, predeclared calibration procedure chose the 95th percentile
        # because it yielded <=3 candidate cases per calibration week.
        calibration_rate=0.05, source_fingerprint=hashlib.sha256(source.read_bytes()).hexdigest())
    calibration_points = scored([(row, at, value) for row, at, value in rows if CAL <= at < EVAL], package)
    calibration_cases = review(calibration_points, package.threshold)
    policy = {"source": SOURCE_KEY, "feature": "temperature", "unit": "source sensor units",
              "budget": LIMIT, "policy": "first three case openings per Monday-Sunday source-clock week",
              "case_rule": "consecutive hourly flags in the same direction; gap, normal reading, or direction change closes a case",
              "calibration_candidate_cases": len(calibration_cases),
              "calibration_queued_cases": sum(case["queued"] for case in calibration_cases)}
    points = scored(later, package)
    cases = review(points, package.threshold)
    out.mkdir(exist_ok=True, parents=True)
    package.save(out / "lens.json")
    save(out / "review-policy.json", policy)
    save(out / "cases.json", cases)
    with (out / "scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=points[0].keys())
        writer.writeheader()
        writer.writerows(points)
    openings = {case["evidence"][0] for case in cases if case["queued"]}
    sample = max((point for point in points if point["source_id"] in openings),
                 key=lambda point: point["score"], default=points[0])
    save(out / "replay.json", replay(out / "lens.json", sample))
    print(json.dumps({"candidate_ref": package.manifest["candidate_ref"],
        "lens_version": package.version, "selection_loss": package.manifest["selection_loss"],
        "threshold": package.threshold, "calibration_flags": package.manifest["calibration_alert_count"],
        "calibration_candidate_cases": len(calibration_cases), "heldout_readings": len(points),
        "candidate_cases": len(cases), "queued_cases": sum(case["queued"] for case in cases)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("series", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    build(args.series, args.out)
