"""Fit a case-level process lens using the reusable REFLEX package builder."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

from reflex.core import build_lens_package, load_lens_package
from reflex.core.models.events import Event
from reflex.core.models.features import EventWindowFeatures
from reflex.experiments.log_adapters import read_bpi2017
from lens import CANDIDATES, FEATURES

OUTCOME_ACTIVITIES = {"A_Cancelled", "A_Denied", "O_Accepted", "O_Cancelled", "O_Refused"}
@dataclass
class _Case:
    case_id: str
    first_event: datetime | None = None
    last_event: datetime | None = None
    first_feature_event: datetime | None = None
    last_feature_event: datetime | None = None
    activities: list[str] = field(default_factory=list)
    resources: set[str] = field(default_factory=set)
    evidence: list[str] = field(default_factory=list)
    outcome_event_count: int = 0
    source_event_count: int = 0

    def add(self, event: Event) -> None:
        self.source_event_count += 1
        at = event.event_time
        self.first_event = at if self.first_event is None else min(self.first_event, at)
        self.last_event = at if self.last_event is None else max(self.last_event, at)
        if event.event_type in OUTCOME_ACTIVITIES:
            self.outcome_event_count += 1
            return
        self.first_feature_event = at if self.first_feature_event is None else min(self.first_feature_event, at)
        self.last_feature_event = at if self.last_feature_event is None else max(self.last_feature_event, at)
        self.activities.append(event.event_type)
        resource = event.attributes.get("org:resource")
        if resource not in (None, ""):
            self.resources.add(str(resource))
        self.evidence.append(event.event_id)

    def row(self) -> EventWindowFeatures:
        assert self.first_event and self.last_event and self.first_feature_event and self.last_feature_event
        activity_counts = {activity: self.activities.count(activity) for activity in set(self.activities)}
        duration = (self.last_feature_event - self.first_feature_event).total_seconds() / 3600
        values = {
            "event_count": float(len(self.activities)),
            "duration_hours": float(duration),
            "activity_count": float(len(activity_counts)),
            "resource_count": float(len(self.resources)),
            "offer_count": float(sum(activity == "O_Create Offer" for activity in self.activities)),
            "rework_count": float(sum(max(0, count - 1) for count in activity_counts.values())),
        }
        # Observation time includes final status completion, but the final
        # accepted/refused/cancelled activity never becomes a feature or evidence.
        if self.last_event <= self.first_event:
            end = self.first_event + timedelta(microseconds=1)
        else:
            end = self.last_event
        return EventWindowFeatures(subject=self.case_id, window_start=self.first_event,
            window_end=end, feature_values=values, supporting_event_ids=tuple(self.evidence),
            event_count=len(self.evidence), cohort="completed-application")


def collect_cases(path: Path) -> tuple[list[EventWindowFeatures], tuple[int, int]]:
    cases: dict[str, _Case] = {}
    for event in read_bpi2017(path):
        case_id = event.entity_ids[0]
        cases.setdefault(case_id, _Case(case_id)).add(event)
    rows = [case.row() for case in cases.values() if case.evidence]
    event_counts = (sum(case.source_event_count for case in cases.values()),
                    sum(case.outcome_event_count for case in cases.values()))
    return sorted(rows, key=lambda row: (row.window_end, row.subject)), event_counts


def temporal_blocks(rows):
    """Cut only between completion-time groups to keep simultaneous cases together."""
    groups = []
    for row in rows:
        if not groups or groups[-1][0].window_end != row.window_end:
            groups.append([])
        groups[-1].append(row)
    total = len(rows)

    def take_boundary(target, start_group=0):
        count = sum(len(group) for group in groups[:start_group])
        for index in range(start_group, len(groups)):
            if count + len(groups[index]) >= target:
                return index + 1, count + len(groups[index])
            count += len(groups[index])
        return len(groups), total

    train_group, train_count = take_boundary(round(total * 0.564))
    cal_group, cal_count = take_boundary(round(total * 0.809), train_group)
    train = [row for group in groups[:train_group] for row in group]
    calibration = [row for group in groups[train_group:cal_group] for row in group]
    test = [row for group in groups[cal_group:] for row in group]
    fit_group, fit_count = take_boundary(round(train_count * 0.8))
    fit_rows = [row for group in groups[:fit_group] for row in group]
    selection_rows = train[fit_count:]
    return fit_rows, selection_rows, calibration, test


def build(source: Path, out: Path):
    rows, (source_event_count, outcome_event_count) = collect_cases(source)
    fit_rows, selection_rows, calibration_rows, test_rows = temporal_blocks(rows)
    package = build_lens_package(
        fit_rows=fit_rows, selection_rows=selection_rows, calibration_rows=calibration_rows,
        candidates=CANDIDATES,
        goal="Review completed applications whose process profile differs from historical cases",
        calibration_rate=0.02, source_fingerprint=hashlib.sha256(source.read_bytes()).hexdigest())
    out.mkdir(parents=True, exist_ok=True)
    package.save(out / "lens.json")
    # Restore through the public package loader to demonstrate that examples
    # supply candidate code and the reusable core handles package integrity.
    package = load_lens_package(out / "lens.json", CANDIDATES)
    observations = package.score(test_rows)
    fields = ["case_id", "completion_time", "score", "threshold", "is_anomaly", "cluster",
              *FEATURES, "supporting_event_count", "evidence_ids"]
    with (out / "case-scores.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row, observation in zip(test_rows, observations, strict=True):
            value = dict(observation.value)
            writer.writerow({"case_id": row.subject, "completion_time": row.window_end.isoformat(),
                "score": value["score"], "threshold": value["threshold"],
                "is_anomaly": value["is_anomaly"], "cluster": value["cluster"],
                "event_count": row.event_count,
                **row.feature_values,
                "supporting_event_count": len(row.supporting_event_ids),
                "evidence_ids": json.dumps(row.supporting_event_ids)})
    summary = {
        "dataset": "BPI Challenge 2017",
        "scope": "completed application case review; post-completion outlier scores",
        "cases": len(rows), "source_events": source_event_count,
        "events_used_in_features": sum(row.event_count for row in rows),
        "outcome_status_events_excluded": outcome_event_count,
        "fit_cases": len(fit_rows), "selection_cases": len(selection_rows),
        "calibration_cases": len(calibration_rows), "test_cases": len(test_rows),
        "candidate_ref": package.manifest["candidate_ref"], "lens_version": package.version,
        "selection_metric": package.manifest["selection_metric"],
        "candidate_selection": package.manifest["candidate_selection"],
        "threshold": package.threshold, "calibration_flags": package.manifest["calibration_alert_count"],
        "test_flags": sum(observation.value["is_anomaly"] for observation in observations),
        "outcome_activities_excluded": sorted(OUTCOME_ACTIVITIES),
        "label_status": "The source log contains outcomes but no independent anomaly labels; no precision/recall claim is made.",
        "top_cases": [{"case_id": row.subject, "completion_time": row.window_end.isoformat(),
                       "score": observation.value["score"], "cluster": observation.value["cluster"],
                       "features": dict(row.feature_values),
                       "evidence_count": len(row.supporting_event_ids)}
                      for row, observation in sorted(zip(test_rows, observations, strict=True),
                                                     key=lambda pair: pair[1].value["score"], reverse=True)[:20]],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: summary[key] for key in (
        "cases", "source_events", "events_used_in_features", "outcome_status_events_excluded",
        "fit_cases", "selection_cases", "calibration_cases", "test_cases",
        "candidate_ref", "threshold", "calibration_flags", "test_flags", "label_status")}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xes_gz", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    build(args.xes_gz, args.out)
