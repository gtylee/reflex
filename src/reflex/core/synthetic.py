"""Seeded synthetic events with separately stored hidden mechanism labels."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .models.events import Event


@dataclass(frozen=True)
class SyntheticDataset:
    events: list[Event]
    ground_truth: list[dict[str, str]]
    start: datetime
    train_end: datetime
    calibration_end: datetime
    end: datetime
    seed: int


def generate_synthetic(seed: int = 11, entities: int = 80, days: int = 30, events_per_day: int = 3) -> SyntheticDataset:
    if entities < 12 or days < 25 or events_per_day < 1:
        raise ValueError("use at least 12 entities, 25 days, and one event per day")
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    train_end = start + timedelta(days=15)
    calibration_end = start + timedelta(days=20)
    end = start + timedelta(days=days)
    event_types = ["open", "update", "review", "approve", "close"]
    entities_meta = {
        f"acct-{i:04d}": {"region": f"region-{i % 4}", "cohort": f"cohort-{i % 8}", "baseline": float(rng.lognormal(5.5, 0.55))}
        for i in range(entities)
    }
    # Two mechanisms are held out for later batches; neither label enters event attributes.
    shared_cause = {f"acct-{i:04d}" for i in range(entities) if 10 <= i < max(14, entities // 5)}
    novel_cohort = {f"acct-{i:04d}" for i in range(entities) if entities // 2 <= i < entities // 2 + max(4, entities // 12)}
    events: list[Event] = []
    truth: list[dict[str, str]] = []
    previous_type: dict[str, str] = {}
    for day in range(days):
        for entity, meta in entities_meta.items():
            # A broad, benign process regime shift begins after the training period.
            regime_multiplier = 1.3 if day >= 15 else 1.0
            count = max(1, int(rng.poisson(events_per_day * regime_multiplier)))
            # A common-cause behaviour starts after the training cutoff.
            shared = day >= 20 and entity in shared_cause
            # A distinct mechanism appears only in the final held-out period.
            novel = day >= 25 and entity in novel_cohort
            for sequence in range(count):
                prior_type = previous_type.get(entity, "open")
                transitions = {
                    "open": ["update", "update", "review"],
                    "update": ["update", "review", "approve"],
                    "review": ["review", "approve", "update"],
                    "approve": ["close", "update"],
                    "close": ["open", "review"],
                }
                event_type = py_rng.choice(transitions.get(prior_type, event_types))
                if rng.random() < 0.006:
                    event_type = "reopen"  # rare, ordinary event sequence element
                hour = int(rng.integers(7, 23))
                minute = int(rng.integers(0, 60))
                event_time = start + timedelta(days=day, hours=hour, minutes=minute, seconds=sequence)
                amount = float(max(1.0, rng.lognormal(5.2, 0.7)))
                # A region/amount association is deliberately spurious and vanishes at the regime shift.
                if meta["region"] == "region-0" and day < 15:
                    amount *= 1.35
                channel = py_rng.choice(["web", "mobile", "branch"])
                latency = float(rng.gamma(2.2, 8.0))
                if shared and sequence == 0:
                    event_type = "approve"
                    channel = "manual_override"
                    latency = float(rng.uniform(0.2, 1.2))
                    amount *= 7.0
                if novel and sequence == 1:
                    # Relationship change: high value and new channel, without extreme standalone amount.
                    amount *= 2.1
                    channel = "partner_api"
                    event_type = "review"
                event_id = f"s{seed}-d{day:03d}-{entity}-{sequence:02d}"
                event = Event(
                    event_id=event_id,
                    event_type=event_type,
                    event_time=event_time,
                    entity_ids=(entity,),
                    attributes={
                        "amount": round(amount, 2), "channel": channel,
                        "latency_minutes": round(latency, 3), "region": meta["region"],
                        "cohort": meta["cohort"], "sequence": sequence,
                    },
                    source="synthetic-v1",
                    provenance={"generator_seed": seed, "generator_version": "1"},
                    relationships=(meta["cohort"], meta["region"]),
                )
                events.append(event)
                previous_type[entity] = event_type
                if shared and sequence == 0:
                    truth.append({"event_id": event_id, "entity_id": entity, "mechanism": "correlated_manual_approval"})
                elif novel and sequence == 1:
                    truth.append({"event_id": event_id, "entity_id": entity, "mechanism": "partner_value_relationship_shift"})
    events.sort(key=lambda event: (event.event_time, event.event_id))
    return SyntheticDataset(events, truth, start, train_end, calibration_end, end, seed)


def write_dataset(dataset: SyntheticDataset, directory: str | Path,
                  evaluator_dir: str | Path | None = None) -> dict[str, str]:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    evaluator_root = Path(evaluator_dir) if evaluator_dir else root.parent / f"{root.name}-evaluator"
    evaluator_root.mkdir(parents=True, exist_ok=True)
    truth_path = evaluator_root / "private_ground_truth.json"
    manifest_path = root / "dataset.json"
    manifest = {
        "dataset": "synthetic-v1", "seed": dataset.seed,
        "event_count": len(dataset.events), "entity_count": len({e.entity_ids[0] for e in dataset.events}),
        "start": dataset.start.isoformat(), "train_end": dataset.train_end.isoformat(),
        "calibration_end": dataset.calibration_end.isoformat(), "end": dataset.end.isoformat(),
        "ground_truth_note": "Hidden labels are stored in a separate evaluator directory; this manifest does not disclose their path.",
    }
    truth_path.write_text(json.dumps(dataset.ground_truth, indent=2) + "\n")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {"manifest": str(manifest_path), "hidden_labels_separate": "true"}

