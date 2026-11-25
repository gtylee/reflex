"""Compare an already saved review queue with the NAB windows for this series."""

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path


def read_clock(text):
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def evaluate(out, labels):
    with (out / "scores.csv").open(newline="") as stream:
        points = list(csv.DictReader(stream))
    times = {point["source_id"]: read_clock(point["time"]) for point in points}
    first, last = min(times.values()), max(times.values())
    source_windows = json.loads(labels.read_text())["realKnownCause/nyc_taxi.csv"]
    windows = [(max(first, read_clock(a)), min(last, read_clock(b))) for a, b in source_windows
               if read_clock(b) >= first and read_clock(a) <= last]
    cases = json.loads((out / "cases.json").read_text())
    covered = set()
    matching = 0
    for case in cases:
        hits = {i for i, (a, b) in enumerate(windows)
                if any(a <= times[event_id] <= b for event_id in case["evidence"])}
        matching += bool(hits)
        covered.update(hits)
    return {"eligible_windows": len(windows), "windows_with_flag": len(covered),
            "cases_overlapping_windows": matching, "cases_outside_windows": len(cases) - matching}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("labels", type=Path)
    args = parser.parse_args()
    report = json.dumps(evaluate(args.out, args.labels), indent=2)
    (args.out / "evaluation.json").write_text(report + "\n")
    print(report)
