"""Independent synthetic-ground-truth scoring and honest manual benchmark records."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .models.lenses import Observation


def score_synthetic_run(observations: list[Observation], truth: list[dict[str, str]], event_times: dict[str, datetime]) -> dict[str, Any]:
    predicted = [observation for observation in observations
                 if isinstance(observation.value, dict) and observation.value.get("is_anomaly")]
    matched_events: set[str] = set()
    matched_mechanisms: set[str] = set()
    true_observations = 0
    for observation in predicted:
        matched = [item for item in truth if item["entity_id"] == observation.subject
                   and observation.window_start <= event_times[item["event_id"]] < observation.window_end]
        if matched:
            true_observations += 1
            matched_events.update(item["event_id"] for item in matched)
            matched_mechanisms.update(item["mechanism"] for item in matched)
    total_truth = len(truth)
    return {
        "predicted_anomalous_observations": len(predicted),
        "true_positive_observations": true_observations,
        "false_discovery_rate": (len(predicted) - true_observations) / len(predicted) if predicted else 0.0,
        "hidden_events_recovered": len(matched_events),
        "hidden_event_recall": len(matched_events) / total_truth if total_truth else 0.0,
        "mechanisms_recovered": sorted(matched_mechanisms),
        "mechanisms_total": sorted({item["mechanism"] for item in truth}),
        "mechanism_recovery_count": len(matched_mechanisms),
        "truth_event_count": total_truth,
    }


def mean_interval(values: list[float], *, confidence: float = 0.95) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "mean": 0.0, "lower": 0.0, "upper": 0.0}
    samples = np.asarray(values, dtype=float)
    rng = np.random.default_rng(8128)
    boot = np.mean(rng.choice(samples, size=(5000, len(samples)), replace=True), axis=1)
    tail = (1 - confidence) / 2
    return {"n": len(values), "mean": float(samples.mean()),
            "lower": float(np.quantile(boot, tail)), "upper": float(np.quantile(boot, 1 - tail))}


VARIANTS = {
    "A": "General-purpose agent with raw events",
    "B": "General-purpose agent with raw events and structured persistent state",
    "C": "General-purpose agent with raw events and shared statistical observations",
    "D": "REFLEX with shared statistical lenses, persistent state, policy and curation",
    "E": "Unrestricted general-purpose agent with persistent workspace and self-built infrastructure",
}


def protocol_manifest(*, seeds: list[int], objective: str, inference_budget: int, compute_budget_seconds: int) -> dict[str, Any]:
    return {
        "protocol_version": 1,
        "objective": objective,
        "seeds": seeds,
        "batches": ["one-shot", "batch-1", "batch-2", "batch-3", "batch-4", "batch-5", "batch-6"],
        "equal_budgets": {"inference_tokens": inference_budget, "non_inference_compute_seconds": compute_budget_seconds},
        "shared_inputs": {"event_batches": "same bytes and order for all variants", "ground_truth": "held by evaluator; never in agent workspace", "statistical_artifacts": "same train cutoff, scaler, model weights, features and calibration threshold for C and D"},
        "variants": [{"id": key, "description": value} for key, value in VARIANTS.items()],
        "run_record_required": ["variant", "seed", "batch", "input_hash", "objective", "output", "inference_tokens", "inference_cost", "compute_seconds", "investigation_count", "evidence_refs", "workspace_ref"],
        "records": [],
        "note": "No experiment result is implied by this manifest. Add one independently recorded run JSON object to records per variant, seed, and batch.",
    }


def append_run_record(manifest_path: str | Path, record: dict[str, Any]) -> None:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    required = manifest["run_record_required"]
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError("run record missing required fields: " + ", ".join(missing))
    if record["variant"] not in VARIANTS:
        raise ValueError("variant must be A, B, C, D, or E")
    if record["seed"] not in manifest["seeds"]:
        raise ValueError("seed is not part of the protocol")
    if record["batch"] not in manifest["batches"]:
        raise ValueError("batch is not part of the protocol")
    if any((item["variant"], item["seed"], item["batch"]) == (record["variant"], record["seed"], record["batch"]) for item in manifest["records"]):
        raise ValueError("a run for this variant, seed, and batch is already recorded")
    if record["inference_tokens"] > manifest["equal_budgets"]["inference_tokens"]:
        raise ValueError("record exceeds the inference token budget")
    if record["compute_seconds"] > manifest["equal_budgets"]["non_inference_compute_seconds"]:
        raise ValueError("record exceeds the compute budget")
    manifest["records"].append(record)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def protocol_report(manifest_path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(manifest_path).read_text())
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in manifest["records"]:
        grouped[record["variant"]].append(record)
    output = []
    for variant, description in VARIANTS.items():
        records = grouped[variant]
        output.append({
            "variant": variant, "description": description, "recorded_runs": len(records),
            "validated_discoveries": sum(int(r["output"].get("validated_discoveries", 0)) for r in records),
            "inference_tokens": sum(int(r["inference_tokens"]) for r in records),
            "inference_cost": sum(float(r["inference_cost"]) for r in records),
            "compute_seconds": sum(float(r["compute_seconds"]) for r in records),
            "investigations": sum(int(r["investigation_count"]) for r in records),
        })
    return {"protocol_version": manifest["protocol_version"], "objective": manifest["objective"], "variants": output,
            "complete": all(len(grouped[key]) == len(manifest["seeds"]) * len(manifest["batches"]) for key in VARIANTS),
            "recorded_runs": len(manifest["records"])}
