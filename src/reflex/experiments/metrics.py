"""Truth-scored metrics with deterministic uncertainty summaries."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from .models.inputs import GroundTruthLabel


def score_predictions(predicted_event_ids: Iterable[str], truth: Iterable[GroundTruthLabel],
                     evidence_refs: dict[str, tuple[str, ...]] | None = None,
                     *, universe_event_ids: Iterable[str] | None = None,
                     events_by_id: dict[str, Any] | None = None,
                     bootstrap_seed: int = 8128, bootstrap_resamples: int = 500) -> dict[str, Any]:
    predicted = set(predicted_event_ids)
    labels = tuple(truth)
    truth_ids = {item.event_id for item in labels}
    tp = len(predicted & truth_ids)
    fp = len(predicted - truth_ids)
    fn = len(truth_ids - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    mechanisms = {item.mechanism for item in labels}
    recovered = {item.mechanism for item in labels if item.event_id in predicted}
    evidence_refs = evidence_refs or {}
    evidence_complete = sum(bool(evidence_refs.get(event_id)) for event_id in predicted)
    result = {
        "precision": precision, "recall": recall,
        "f1": (2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "predicted_events": len(predicted), "truth_events": len(truth_ids),
        "mechanisms_recovered": sorted(recovered), "mechanisms_total": sorted(mechanisms),
        "mechanism_recall": len(recovered) / len(mechanisms) if mechanisms else 0.0,
        "evidence_completeness": evidence_complete / len(predicted) if predicted else 1.0,
    }
    if events_by_id is not None:
        reference_count = 0
        known_count = 0
        temporal_count = 0
        entity_count = 0
        integrity_count = 0
        target_inclusions = 0
        supported_predictions = 0
        for event_id in predicted:
            target = events_by_id.get(event_id)
            references = tuple(evidence_refs.get(event_id, ()))
            reference_count += len(references)
            if event_id in references:
                target_inclusions += 1
            if not references:
                continue
            all_supported = target is not None
            for reference_id in references:
                reference = events_by_id.get(reference_id)
                known = reference is not None and target is not None
                temporal = known and reference.event_time <= target.event_time
                target_entities = set(getattr(target, "entity_ids", ()) or ()) if target else set()
                reference_entities = set(getattr(reference, "entity_ids", ()) or ()) if reference else set()
                same_entity = known and bool(target_entities & reference_entities)
                integrity = temporal and same_entity
                known_count += int(known)
                temporal_count += int(temporal)
                entity_count += int(same_entity)
                integrity_count += int(integrity)
                all_supported = all_supported and integrity
            supported_predictions += int(all_supported)
        no_references_score = 1.0 if not predicted else 0.0
        result.update({
            "evidence_reference_count": reference_count,
            "evidence_reference_integrity": known_count / reference_count if reference_count else no_references_score,
            "evidence_temporal_validity": temporal_count / reference_count if reference_count else no_references_score,
            "evidence_entity_alignment": entity_count / reference_count if reference_count else no_references_score,
            "evidence_integrity": integrity_count / reference_count if reference_count else no_references_score,
            "evidence_target_inclusion": target_inclusions / len(predicted) if predicted else 1.0,
            "evidence_supported_prediction_rate": supported_predictions / len(predicted) if predicted else 1.0,
        })
    if labels and bootstrap_resamples > 0:
        rng = np.random.default_rng(bootstrap_seed)
        universe = sorted(set(universe_event_ids or ()) | truth_ids | predicted)
        truth_mask = np.asarray([event_id in truth_ids for event_id in universe], dtype=bool)
        pred_mask = np.asarray([event_id in predicted for event_id in universe], dtype=bool)
        sampled_f1 = []
        for indices in rng.integers(0, len(truth_mask), size=(bootstrap_resamples, len(truth_mask))):
            y_true, y_pred = truth_mask[indices], pred_mask[indices]
            tp_sample = int(np.sum(y_true & y_pred))
            fp_sample = int(np.sum(~y_true & y_pred))
            fn_sample = int(np.sum(y_true & ~y_pred))
            p = tp_sample / (tp_sample + fp_sample) if tp_sample + fp_sample else 0.0
            r = tp_sample / (tp_sample + fn_sample) if tp_sample + fn_sample else 0.0
            sampled_f1.append(2 * p * r / (p + r) if p + r else 0.0)
        result["f1_interval_95"] = {"lower": float(np.quantile(sampled_f1, 0.025)),
                                    "upper": float(np.quantile(sampled_f1, 0.975)),
                                    "resamples": bootstrap_resamples, "seed": bootstrap_seed}
    else:
        result["f1_interval_95"] = None
    return result


def mean_interval(values: list[float], *, confidence: float = 0.95, seed: int = 8128,
                  resamples: int = 5000) -> dict[str, float | int]:
    if not 0 < confidence < 1 or resamples < 1:
        raise ValueError("confidence must be between 0 and 1 and resamples must be positive")
    if not values:
        return {"n": 0, "mean": 0.0, "lower": 0.0, "upper": 0.0}
    samples = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.mean(rng.choice(samples, size=(resamples, len(samples)), replace=True), axis=1)
    tail = (1 - confidence) / 2
    return {"n": len(values), "mean": float(samples.mean()),
            "lower": float(np.quantile(boot, tail)), "upper": float(np.quantile(boot, 1 - tail))}


def decision_variance(decisions: Iterable[Any], correct_decision: Any | None = None) -> dict[str, float | int | None]:
    values = list(decisions)
    if not values:
        return {"runs": 0, "unique_outcomes": 0, "disagreement_rate": 0.0,
                "numerical_variance": None, "accuracy": None}
    counts: dict[str, int] = {}
    for item in values:
        key = repr(item)
        counts[key] = counts.get(key, 0) + 1
    majority = max(counts.values()) / len(values)
    numeric = all(isinstance(value, (int, float, np.number)) and not isinstance(value, bool) for value in values)
    accuracy = (sum(value == correct_decision for value in values) / len(values)
                if correct_decision is not None else None)
    return {"runs": len(values), "unique_outcomes": len(counts), "disagreement_rate": 1 - majority,
            "numerical_variance": float(np.var(values)) if numeric else None, "accuracy": accuracy}


__all__ = ["score_predictions", "mean_interval", "decision_variance"]
