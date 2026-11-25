"""Training-only lens selection for event streams.

The selector uses only training events to inspect the stream shape and choose a
candidate lens. It calibrates a score cutoff on the declared calibration slice.
Evaluation labels stay outside this module.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
from math import log
from typing import Any

import numpy as np

from .models.inputs import AgentInput
from ..registry import DetectorRegistry


CALIBRATION_QUANTILE = 0.99


@dataclass(frozen=True)
class LensSelection:
    """A lens choice and score bundle derived from one visible input stream."""

    lens_id: str
    rationale: str
    features: tuple[str, ...]
    training_profile: dict[str, Any]
    threshold: float
    calibration_alert_count: int
    scores: dict[str, float]
    lens_version: str | None = None
    provider: str | None = None
    provider_version: str | None = None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.number)) and not isinstance(value, bool)


def _attribute_profile(events) -> tuple[tuple[str, ...], tuple[str, ...], dict[str, float]]:
    numeric_values: dict[str, list[float]] = defaultdict(list)
    categorical_values: dict[str, list[str]] = defaultdict(list)
    for event in events:
        for name, value in event.attributes.items():
            if _is_number(value) and np.isfinite(float(value)):
                numeric_values[name].append(float(value))
            elif isinstance(value, str) and value:
                categorical_values[name].append(value)

    numeric_fields = tuple(sorted(
        name for name, values in numeric_values.items()
        if len(values) >= 12 and len(set(values)) > 1
    ))
    categorical_fields = tuple(sorted(
        name for name, values in categorical_values.items() if len(values) >= 12
    ))
    singleton_categories = {
        name: float(len(set(categorical_values[name])))
        for name in categorical_fields if len(set(categorical_values[name])) == 1
    }
    return numeric_fields, categorical_fields, singleton_categories


def _robust_center_scale(values: np.ndarray) -> tuple[float, float]:
    center = float(np.median(values))
    deviations = np.abs(values - center)
    mad_scale = float(1.4826 * np.median(deviations))
    q80_scale = float(np.quantile(deviations, 0.80) / 1.2816)
    q25, q75 = np.quantile(values, [0.25, 0.75])
    iqr_scale = float((q75 - q25) / 1.349)
    scale = max(mad_scale, q80_scale, iqr_scale, 1e-6)
    return center, scale


def _correlation_profile(events, numeric_fields: tuple[str, ...]) -> tuple[float, tuple[str, str] | None]:
    strongest = 0.0
    strongest_pair: tuple[str, str] | None = None
    for left_index, left in enumerate(numeric_fields):
        for right in numeric_fields[left_index + 1:]:
            pairs = [(float(event.attributes[left]), float(event.attributes[right]))
                     for event in events
                     if left in event.attributes and right in event.attributes
                     and _is_number(event.attributes[left]) and _is_number(event.attributes[right])]
            if len(pairs) < 20:
                continue
            matrix = np.asarray(pairs, dtype=float)
            if np.std(matrix[:, 0]) < 1e-10 or np.std(matrix[:, 1]) < 1e-10:
                continue
            correlation = float(np.corrcoef(matrix[:, 0], matrix[:, 1])[0, 1])
            if abs(correlation) > strongest:
                strongest = abs(correlation)
                strongest_pair = (left, right)
    return strongest, strongest_pair


def _training_profile(events, numeric_fields: tuple[str, ...], categorical_fields: tuple[str, ...]) -> dict[str, Any]:
    entity_days: dict[tuple[str, str], int] = Counter()
    for event in events:
        entity = event.entity_ids[0] if event.entity_ids else "unassigned"
        entity_days[(entity, event.event_time.date().isoformat())] += 1
    mean_events_per_entity_day = (float(np.mean(list(entity_days.values()))) if entity_days else 0.0)
    max_correlation, pair = _correlation_profile(events, numeric_fields)
    event_type_count = len({event.event_type for event in events})
    return {
        "training_event_count": len(events),
        "numeric_fields": list(numeric_fields),
        "categorical_fields": list(categorical_fields),
        "event_type_count": event_type_count,
        "mean_events_per_entity_day": mean_events_per_entity_day,
        "max_abs_numeric_correlation": max_correlation,
        "strongest_numeric_pair": list(pair) if pair else None,
    }


def _choose_lens(profile: dict[str, Any], singleton_categories: dict[str, float]) -> tuple[str, str, tuple[str, ...]]:
    numeric_fields = tuple(profile["numeric_fields"])
    categorical_fields = tuple(profile["categorical_fields"])
    if profile["mean_events_per_entity_day"] >= 1.5 and profile["event_type_count"] >= 5:
        return (
            "sequence_novelty",
            "Training has repeated multi-event entity days and at least five event types, so REFLEX selected an ordered-sequence lens.",
            (),
        )
    if singleton_categories and numeric_fields:
        field = sorted(singleton_categories)[0]
        return (
            "categorical_novelty",
            f"Training has a numeric stream plus the constant category {field!r}, so REFLEX selected a category-change lens.",
            (field,),
        )
    if len(numeric_fields) >= 3 or (
        len(numeric_fields) >= 2 and profile["max_abs_numeric_correlation"] >= 0.45
    ):
        pair = profile["strongest_numeric_pair"]
        detail = f" The strongest training relationship is {pair[0]} with {pair[1]}." if pair else ""
        return (
            "multivariate_relationship",
            f"Training has {len(numeric_fields)} varying numeric fields, so REFLEX selected a joint-distribution lens.{detail}",
            numeric_fields,
        )
    if numeric_fields:
        return (
            "robust_numeric_drift",
            f"Training is dominated by {len(numeric_fields)} varying numeric field(s) without a strong multivariate relationship, so REFLEX selected a robust numeric drift lens.",
            numeric_fields,
        )
    if categorical_fields:
        return (
            "categorical_novelty",
            "Training has categorical attributes but no varying numeric fields, so REFLEX selected a category novelty lens.",
            categorical_fields,
        )
    return "event_type_novelty", "Training contains no usable varying attributes, so REFLEX selected event-type novelty.", (),


def _numeric_values(events, numeric_fields: tuple[str, ...], medians: dict[str, float]) -> np.ndarray:
    return np.asarray([
        [float(event.attributes[name]) if name in event.attributes and _is_number(event.attributes[name])
         else medians[name] for name in numeric_fields]
        for event in events
    ], dtype=float)


def _numeric_statistics(training_events, numeric_fields: tuple[str, ...]):
    medians: dict[str, float] = {}
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name in numeric_fields:
        values = np.asarray([float(event.attributes[name]) for event in training_events
                             if name in event.attributes and _is_number(event.attributes[name])], dtype=float)
        median = float(np.median(values))
        center, scale = _robust_center_scale(values)
        medians[name] = median
        centers[name] = center
        scales[name] = scale
    return medians, centers, scales


def _univariate_scores(events, numeric_fields, centers, scales, *, use_cusum: bool) -> np.ndarray:
    scores = np.zeros(len(events), dtype=float)
    entity_cusum: dict[tuple[str, str], tuple[float, float]] = {}
    order = sorted(range(len(events)), key=lambda index: (
        events[index].entity_ids[0] if events[index].entity_ids else "unassigned",
        events[index].event_time,
        events[index].event_id,
    ))
    for index in order:
        event = events[index]
        entity = event.entity_ids[0] if event.entity_ids else "unassigned"
        for name in numeric_fields:
            value = event.attributes.get(name)
            if not _is_number(value):
                continue
            z_score = (float(value) - centers[name]) / scales[name]
            score = abs(z_score)
            if use_cusum:
                key = (entity, name)
                positive, negative = entity_cusum.get(key, (0.0, 0.0))
                positive = max(0.0, positive + z_score - 0.5)
                negative = max(0.0, negative - z_score - 0.5)
                entity_cusum[key] = (positive, negative)
                score = max(score, positive, negative)
            scores[index] = max(scores[index], score)
    return scores


def _relationship_scores(events, training_events, numeric_fields) -> np.ndarray:
    if not numeric_fields:
        return np.zeros(len(events), dtype=float)
    medians, centers, scales = _numeric_statistics(training_events, numeric_fields)
    train_matrix = _numeric_values(training_events, numeric_fields, medians)
    all_matrix = _numeric_values(events, numeric_fields, medians)
    train_z = (train_matrix - np.asarray([centers[name] for name in numeric_fields])) / np.asarray(
        [scales[name] for name in numeric_fields])
    all_z = (all_matrix - np.asarray([centers[name] for name in numeric_fields])) / np.asarray(
        [scales[name] for name in numeric_fields])
    covariance = np.cov(train_z, rowvar=False)
    covariance = np.atleast_2d(covariance)
    regularized = covariance + np.eye(covariance.shape[0]) * 0.1
    inverse = np.linalg.pinv(regularized)
    squared_distance = np.einsum("ij,jk,ik->i", all_z, inverse, all_z)
    return np.sqrt(np.maximum(squared_distance, 0.0))


def _categorical_scores(events, training_events, categorical_fields) -> np.ndarray:
    counts: dict[str, Counter[str]] = {name: Counter() for name in categorical_fields}
    totals = Counter()
    for event in training_events:
        for name in categorical_fields:
            value = event.attributes.get(name)
            if isinstance(value, str) and value:
                counts[name][value] += 1
                totals[name] += 1
    scores = np.zeros(len(events), dtype=float)
    for index, event in enumerate(events):
        for name in categorical_fields:
            value = event.attributes.get(name)
            if not isinstance(value, str) or not value:
                continue
            vocabulary_size = max(1, len(counts[name]))
            probability = (counts[name][value] + 0.5) / (totals[name] + 0.5 * (vocabulary_size + 1))
            scores[index] = max(scores[index], -log(probability))
    return scores


def _sequence_scores(events, training_events, *, max_gap: timedelta = timedelta(hours=1)) -> np.ndarray:
    counts: Counter[tuple[str, ...]] = Counter()
    train_total = 0
    by_entity: dict[str, list[Any]] = defaultdict(list)
    for event in training_events:
        entity = event.entity_ids[0] if event.entity_ids else "unassigned"
        by_entity[entity].append(event)
    for entity_events in by_entity.values():
        ordered = sorted(entity_events, key=lambda item: (item.event_time, item.event_id))
        for index in range(3, len(ordered)):
            sequence = ordered[index - 3:index + 1]
            if sequence[-1].event_time - sequence[0].event_time <= max_gap:
                counts[tuple(item.event_type for item in sequence)] += 1
                train_total += 1
    scores = np.zeros(len(events), dtype=float)
    indices_by_entity: dict[str, list[int]] = defaultdict(list)
    for index, event in enumerate(events):
        entity = event.entity_ids[0] if event.entity_ids else "unassigned"
        indices_by_entity[entity].append(index)
    vocabulary_size = max(1, len(counts))
    for indices in indices_by_entity.values():
        ordered_indices = sorted(indices, key=lambda index: (events[index].event_time, events[index].event_id))
        for position in range(3, len(ordered_indices)):
            window = ordered_indices[position - 3:position + 1]
            sequence_events = [events[index] for index in window]
            if sequence_events[-1].event_time - sequence_events[0].event_time > max_gap:
                continue
            pattern = tuple(item.event_type for item in sequence_events)
            probability = (counts[pattern] + 0.5) / (train_total + 0.5 * (vocabulary_size + 1))
            scores[ordered_indices[position]] = -log(probability)
    return scores


def select_lens(experiment: AgentInput, *, registry: DetectorRegistry | None = None) -> LensSelection:
    """Select and calibrate a lens using training data and a held-out calibration slice.

    Installed providers can offer detectors through the
    ``reflex_intelligence.detectors`` entry-point group. Their compatibility
    score and fit methods receive training data only. Once a detector is
    selected, it scores the stream and REFLEX sets its threshold from the
    calibration partition. If no provider is applicable, the built-in lens
    heuristics remain the fallback.
    """
    training_events = [event for event in experiment.events if event.event_time < experiment.train_end]
    calibration_events = [event for event in experiment.events
                          if experiment.train_end <= event.event_time < experiment.calibration_end]
    if len(training_events) < 20 or len(calibration_events) < 10:
        raise ValueError("lens selection needs at least 20 training and 10 calibration events")
    numeric_fields, categorical_fields, singleton_categories = _attribute_profile(training_events)
    profile = _training_profile(training_events, numeric_fields, categorical_fields)
    detector_registry = registry or _discover_detectors()
    candidates = detector_registry.candidates(profile)
    provider = None
    provider_version = None
    lens_version = None
    if candidates:
        match_score, detector = candidates[0]
        training = deepcopy(tuple(training_events))
        fitted = detector.fit(training, deepcopy(profile))
        detector_scores = detector.score(fitted, deepcopy(tuple(experiment.events)))
        if not isinstance(detector_scores, Mapping):
            raise TypeError(f"detector {detector.ref!r} must return scores keyed by event ID")
        expected_ids = {event.event_id for event in experiment.events}
        if set(detector_scores) != expected_ids:
            missing = expected_ids.difference(detector_scores)
            extra = set(detector_scores).difference(expected_ids)
            raise ValueError(f"detector {detector.ref!r} returned mismatched event IDs (missing={len(missing)}, extra={len(extra)})")
        raw_scores = [detector_scores[event.event_id] for event in experiment.events]
        if any(not _is_number(value) for value in raw_scores):
            raise ValueError(f"detector {detector.ref!r} returned a non-numeric score")
        score_values = np.asarray([float(value) for value in raw_scores], dtype=float)
        if not np.isfinite(score_values).all():
            raise ValueError(f"detector {detector.ref!r} returned a non-finite score")
        lens_id = detector.lens_id
        lens_version = detector.version
        rationale = (f"Installed detector {detector.ref} was selected from training data "
                     f"with a compatibility score of {match_score:.3f}. {detector.description}")
        selected_fields = detector.features
        provider = detector.provider
        provider_version = detector.provider_version
        profile["selected_match_score"] = match_score
    else:
        lens_id, rationale, selected_fields = _choose_lens(profile, singleton_categories)
        medians, centers, scales = _numeric_statistics(training_events, numeric_fields)

        if lens_id == "sequence_novelty":
            score_values = _sequence_scores(list(experiment.events), training_events)
        elif lens_id == "categorical_novelty":
            category_fields = selected_fields or categorical_fields
            score_values = _categorical_scores(list(experiment.events), training_events, category_fields)
        elif lens_id == "multivariate_relationship":
            score_values = _relationship_scores(list(experiment.events), training_events, numeric_fields)
        else:
            score_values = _univariate_scores(list(experiment.events), numeric_fields, centers, scales,
                                              use_cusum=(lens_id == "robust_numeric_drift"))

    event_scores = {event.event_id: float(score_values[index])
                    for index, event in enumerate(experiment.events)}
    calibration_scores = np.asarray([event_scores[event.event_id] for event in calibration_events], dtype=float)
    threshold = float(np.quantile(calibration_scores, CALIBRATION_QUANTILE))
    calibration_alert_count = int(np.sum(calibration_scores >= threshold)) if threshold > 0 else int(np.sum(calibration_scores > 0))
    profile["selected_lens"] = lens_id
    profile["selected_lens_version"] = lens_version
    profile["selected_features"] = list(selected_fields)
    profile["selected_provider"] = provider
    profile["selected_provider_version"] = provider_version
    return LensSelection(
        lens_id=lens_id,
        rationale=rationale,
        features=selected_fields,
        training_profile=profile,
        threshold=threshold,
        calibration_alert_count=calibration_alert_count,
        scores=event_scores,
        lens_version=lens_version,
        provider=provider,
        provider_version=provider_version,
    )


def _discover_detectors() -> DetectorRegistry:
    """Load installed providers at selection time so new libraries can compete."""
    registry = DetectorRegistry()
    for name in registry.available_libraries():
        registry.load_library(name)
    return registry


def simple_numeric_baseline(experiment: AgentInput) -> LensSelection:
    """Fit a maximum robust numeric-deviation score as the comparison baseline."""
    training_events = [event for event in experiment.events if event.event_time < experiment.train_end]
    calibration_events = [event for event in experiment.events
                          if experiment.train_end <= event.event_time < experiment.calibration_end]
    numeric_fields, _, _ = _attribute_profile(training_events)
    _, centers, scales = _numeric_statistics(training_events, numeric_fields)
    values = _univariate_scores(list(experiment.events), numeric_fields, centers, scales, use_cusum=False)
    scores = {event.event_id: float(values[index]) for index, event in enumerate(experiment.events)}
    calibration_scores = np.asarray([scores[event.event_id] for event in calibration_events], dtype=float)
    threshold = float(np.quantile(calibration_scores, CALIBRATION_QUANTILE))
    calibration_alert_count = int(np.sum(calibration_scores >= threshold)) if threshold > 0 else int(np.sum(calibration_scores > 0))
    return LensSelection(
        lens_id="robust_numeric_baseline",
        rationale="Maximum robust deviation across numeric training fields, calibrated at the 99th percentile.",
        features=numeric_fields,
        training_profile={"numeric_fields": list(numeric_fields)},
        threshold=threshold,
        calibration_alert_count=calibration_alert_count,
        scores=scores,
    )


__all__ = ["LensSelection", "select_lens", "simple_numeric_baseline"]
