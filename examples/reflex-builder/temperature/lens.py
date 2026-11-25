"""NAB temperature candidates and their REFLEX Reaction binding.

Peer construction and the meaning of temperature deviations live under
``examples``. The core sees only named numeric features and evidence IDs.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
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

LENS_ID = "office.temperature-deviation"
EVENT_TYPE = "office.temperature"


def slot(at: datetime) -> str:
    return f"{'weekend' if at.weekday() >= 5 else 'weekday'}-{at.hour:02d}"


def _baseline(values: Sequence[float]) -> dict[str, float | int]:
    center = median(values)
    return {"center": center, "scale": 1.4826 * median(abs(x - center) for x in values),
            "training_rows": len(values)}


def _fit(rows: Sequence[EventWindowFeatures], peers: bool) -> dict[str, Any]:
    values = [row.feature_values["temperature"] for row in rows]
    pooled = _baseline(values)
    if pooled["scale"] == 0:
        raise ValueError("pooled training temperatures have zero robust spread")
    model: dict[str, Any] = {"slots": {}, "baselines": {"pooled": pooled}}
    if not peers:
        return model
    by_slot: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_slot[slot(row.window_start)].append(row.feature_values["temperature"])
    supported = sorted(key for key, slot_values in by_slot.items() if len(slot_values) >= 7)
    typical = np.array([median(by_slot[key]) for key in supported])
    if len(set(typical)) < 3:
        return model
    groups = KMeans(n_clusters=3, n_init=10, random_state=11).fit(typical.reshape(-1, 1))
    names = dict(zip(np.argsort(groups.cluster_centers_.ravel()), ("cool", "middle", "warm")))
    members: dict[str, list[float]] = defaultdict(list)
    for key, group in zip(supported, groups.labels_, strict=True):
        peer = names[group]
        model["slots"][key] = peer
        members[peer].extend(by_slot[key])
    for name, peer_values in members.items():
        stats = _baseline(peer_values)
        stats["scale"] = stats["scale"] or pooled["scale"]
        model["baselines"][name] = stats
    return model


def compare(model: Mapping[str, Any], at: datetime, value: float) -> dict[str, Any]:
    peer = model["slots"].get(slot(at), "pooled")
    stats = model["baselines"][peer]
    delta = value - stats["center"]
    return {"observed": value, "expected": stats["center"], "scale": stats["scale"],
            "peer": peer, "direction": "high" if delta >= 0 else "low",
            "score": abs(delta) / stats["scale"]}


class TemperatureCandidate:
    """One training-only comparator for an unusual hourly reading."""

    lens_id = LENS_ID
    feature_names = ("temperature",)
    metric_name = "median_absolute_error"
    provider = "examples.reflex-builder.temperature"
    provider_version = "1"

    def __init__(self, *, peers: bool):
        self.peers = peers
        self.version = "peer-v1" if peers else "pooled-v1"

    def fit(self, rows: Sequence[EventWindowFeatures]) -> dict[str, Any]:
        return _fit(rows, self.peers)

    def validation_loss(self, model: Mapping[str, Any], rows: Sequence[EventWindowFeatures]) -> float:
        return median(abs(row.feature_values["temperature"] -
                          compare(model, row.window_start, row.feature_values["temperature"])["expected"])
                      for row in rows)

    def score(self, model: Mapping[str, Any], rows: Sequence[EventWindowFeatures]) -> list[float]:
        return [compare(model, row.window_start, row.feature_values["temperature"])["score"] for row in rows]

    def explain(self, model: Mapping[str, Any], row: EventWindowFeatures) -> Mapping[str, Any]:
        details = compare(model, row.window_start, row.feature_values["temperature"])
        details.pop("score")
        return details

    def encode(self, model: Mapping[str, Any]) -> Mapping[str, Any]:
        return model

    def decode(self, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        return dict(payload)


POOLED = TemperatureCandidate(peers=False)
PEERS = TemperatureCandidate(peers=True)
CANDIDATES = (POOLED, PEERS)


def feature_row(at: datetime, value: float, source_id: str) -> EventWindowFeatures:
    return EventWindowFeatures(subject="office-sensor", window_start=at,
        window_end=at + timedelta(hours=1), feature_values={"temperature": value},
        supporting_event_ids=(source_id,), event_count=1, cohort="all")


def observe(event, _state, *, package: FittedLensPackage):
    row = feature_row(event.event_time, float(event.attributes["temperature"]), event.event_id)
    observation = package.score([row])[0]
    event_out = observation.as_event()
    return [ObservationEvent(
        **{**event_out.model_dump(), "event_type": "observation.temperature_deviation"})]


def bind(package: FittedLensPackage, registry):
    ref = FunctionRef(function_id=LENS_ID, version=package.version, kind="lens")
    registry.register_many([FunctionSpec(ref, partial(observe, package=package))])
    return ReactionDefinition(reaction_id=LENS_ID, version=ref.version,
                              subscriptions=(EVENT_TYPE,), nodes=(ReactionNode(node_id="score", function=ref),))
