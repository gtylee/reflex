"""NYC taxi demand candidates and their REFLEX Reaction binding.

This file is intentionally under ``examples``: slot definitions, passenger
units, peer clustering, and the interpretation of a quiet half-hour are
properties of this dataset, not of the reusable REFLEX core.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from functools import partial
from statistics import median
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from reflex.contracts.reactions import FunctionRef, ReactionDefinition, ReactionNode
from reflex.core import FittedLensPackage
from reflex.core.models.events import ObservationEvent
from reflex.core.models.features import EventWindowFeatures
from reflex.registry import FunctionSpec

LENS = "nyc-taxi.low-demand"
EVENT = "taxi.half_hour"


def slot(timestamp: datetime) -> str:
    """Represent the source clock as weekday and half-hour slot."""
    return f"{timestamp.weekday()}/{timestamp.hour:02d}:{timestamp.minute:02d}"


def _baseline(values: Sequence[float]) -> dict[str, float | int]:
    center = median(values)
    mad = median(abs(value - center) for value in values)
    return {"median": center, "scale": max(1.0, 1.4826 * mad), "training_rows": len(values)}


def _fit(rows: Sequence[EventWindowFeatures], use_peers: bool) -> dict[str, Any]:
    counts = [row.feature_values["passengers"] for row in rows]
    model: dict[str, Any] = {"baselines": {"pooled": _baseline(counts)}, "slot_peers": {}}
    if not use_peers:
        return model

    by_slot: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_slot[slot(row.window_start)].append(row.feature_values["passengers"])
    supported = sorted(key for key, observations in by_slot.items() if len(observations) >= 4)
    typical = np.array([median(by_slot[key]) for key in supported])
    if len(set(typical)) < 4:
        return model
    clusters = KMeans(n_clusters=4, n_init=10, random_state=11).fit(typical.reshape(-1, 1))
    names = {index: f"demand-{rank}" for rank, index in
             enumerate(np.argsort(clusters.cluster_centers_.ravel()), start=1)}
    members: dict[str, list[float]] = defaultdict(list)
    for key, label in zip(supported, clusters.labels_, strict=True):
        peer = names[label]
        model["slot_peers"][key] = peer
        members[peer].extend(by_slot[key])
    model["baselines"].update({peer: _baseline(values) for peer, values in members.items()})
    return model


def compare(model: Mapping[str, Any], timestamp: datetime, passengers: float) -> dict[str, Any]:
    peer = model["slot_peers"].get(slot(timestamp), "pooled")
    baseline = model["baselines"][peer]
    deficit = float(baseline["median"]) - passengers
    return {"passengers": passengers, "peer": peer, "expected": baseline["median"],
            "scale": baseline["scale"], "score": max(0.0, deficit / baseline["scale"])}


class TaxiDemandCandidate:
    """One training-only comparator for a quiet taxi half-hour."""

    lens_id = LENS
    feature_names = ("passengers",)
    metric_name = "median_absolute_error"
    provider = "examples.reflex-builder.taxi"
    provider_version = "1"

    def __init__(self, *, use_peers: bool):
        self.use_peers = use_peers
        self.version = "peer-v1" if use_peers else "pooled-v1"

    def fit(self, rows: Sequence[EventWindowFeatures]) -> dict[str, Any]:
        return _fit(rows, self.use_peers)

    def validation_loss(self, model: Mapping[str, Any], rows: Sequence[EventWindowFeatures]) -> float:
        return median(abs(row.feature_values["passengers"] -
                          compare(model, row.window_start, row.feature_values["passengers"])["expected"])
                      for row in rows)

    def score(self, model: Mapping[str, Any], rows: Sequence[EventWindowFeatures]) -> list[float]:
        return [compare(model, row.window_start, row.feature_values["passengers"])["score"] for row in rows]

    def explain(self, model: Mapping[str, Any], row: EventWindowFeatures) -> Mapping[str, Any]:
        details = compare(model, row.window_start, row.feature_values["passengers"])
        details.pop("score")
        return details

    def encode(self, model: Mapping[str, Any]) -> Mapping[str, Any]:
        return model

    def decode(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return dict(payload)


POOLED = TaxiDemandCandidate(use_peers=False)
PEERS = TaxiDemandCandidate(use_peers=True)
CANDIDATES = (POOLED, PEERS)


def feature_row(timestamp: datetime, passengers: float, source_id: str) -> EventWindowFeatures:
    """Normalize one source point into the core's generic row contract."""
    from datetime import timedelta

    return EventWindowFeatures(subject="nyc-taxi", window_start=timestamp,
        window_end=timestamp + timedelta(minutes=30), feature_values={"passengers": passengers},
        supporting_event_ids=(source_id,), event_count=1, cohort="all")


def observe(event, _state, *, package: FittedLensPackage):
    row = feature_row(event.event_time, float(event.attributes["passengers"]), event.event_id)
    observation = package.score([row])[0]
    value = dict(observation.value)
    reason = (f"{value['passengers']:g} passengers versus a training median of "
              f"{value['expected']:g} for {value['peer']}.")
    event_out = observation.as_event()
    return [ObservationEvent(
        **{**event_out.model_dump(), "event_type": "observation.taxi_demand",
           "payload": {**event_out.payload, "value": value, "reason": reason}})]


def bind(package: FittedLensPackage, registry):
    """Register a fitted version and return a Reaction executable by a host."""
    ref = FunctionRef(function_id=LENS, version=package.version, kind="lens")
    registry.register_many([FunctionSpec(ref, partial(observe, package=package))])
    return ReactionDefinition(reaction_id=LENS, version=ref.version, subscriptions=(EVENT,),
                              nodes=(ReactionNode(node_id="score", function=ref),))
