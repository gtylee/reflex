"""Audit the frozen weekly queue and count hits against published NAB windows."""

import argparse
import csv
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path


def stamp(text):
    at = datetime.fromisoformat(text)
    return at.replace(tzinfo=UTC) if at.tzinfo is None else at.astimezone(UTC)


def evaluate(out, label_file):
    package = json.loads((out / "lens.json").read_text())
    policy = json.loads((out / "review-policy.json").read_text())
    with (out / "scores.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    times = {row["source_id"]: stamp(row["time"]) for row in rows}
    cases = json.loads((out / "cases.json").read_text())
    first, last = min(times.values()), max(times.values())
    monday = first.date() - timedelta(days=first.weekday())
    weeks = {}
    while monday <= last.date():
        weeks[monday.isoformat()] = {"candidates": 0, "queued": 0, "readings": 0}
        monday += timedelta(days=7)
    for at in times.values():
        weeks[(at.date() - timedelta(days=at.weekday())).isoformat()]["readings"] += 1

    previous_opening = None
    for case in cases:
        opened = times[case["evidence"][0]]
        if previous_opening is not None and opened <= previous_opening:
            raise ValueError("Case openings are out of order")
        previous_opening = opened
        week_key = (opened.date() - timedelta(days=opened.weekday())).isoformat()
        w = weeks[week_key]
        w["candidates"] += 1
        if (week_key != case["week_start"] or stamp(case["opened"]) != opened or
                case["queued"] != (w["candidates"] <= policy["budget"])):
            raise ValueError("Case admission disagrees with its chronological budget slot")
        w["queued"] += int(case["queued"])
    if any(w["queued"] > policy["budget"] for w in weeks.values()):
        raise ValueError("Review budget exceeded")

    windows = []
    case_hits = set()
    for a, b in json.loads(label_file.read_text())[policy["source"]]:
        start, end = stamp(a), stamp(b)
        if end < first or start > last:
            continue
        available = sum(start <= t <= end for t in times.values())
        caught = {"queued": [], "suppressed": []}
        overlap = set()
        for case in cases:
            opened = stamp(case["opened"])
            if start <= opened <= end:
                disposition = "queued" if case["queued"] else "suppressed"
                caught[disposition].append(case["opened"])
                if case["queued"]:
                    case_hits.add(opened)
            if case["queued"] and any(start <= times[e] <= end for e in case["evidence"]):
                overlap.add(opened)
        windows.append({"start": a, "end": b, "available_readings": available,
                        "queued_openings": caught["queued"], "suppressed_openings": caught["suppressed"],
                        "queued_cases_with_evidence_in_window": len(overlap)})
    queued = sum(w["queued"] for w in weeks.values())
    return {"start": first.isoformat(), "end": last.isoformat(), "heldout_readings": len(rows),
            "point_crossings": sum(float(r["score"]) > package["threshold"] for r in rows),
            "calendar_weeks": len(weeks), "weekly_budget": policy["budget"],
            "highest_weekly_workload": max(w["queued"] for w in weeks.values()),
            "candidate_cases": len(cases), "queued_cases": queued, "suppressed_cases": len(cases) - queued,
            "labeled_windows": len(windows), "windows_hit": sum(bool(w["queued_openings"]) for w in windows),
            "windows_hit_without_cap": sum(bool(w["queued_openings"] or w["suppressed_openings"]) for w in windows),
            "queued_cases_opened_in_windows": len(case_hits), "queued_cases_opened_outside": queued - len(case_hits),
            "weekly_counts": weeks, "window_details": windows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("labels", type=Path)
    args = parser.parse_args()
    result = evaluate(args.out, args.labels)
    (args.out / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "weekly_counts"}, indent=2))
