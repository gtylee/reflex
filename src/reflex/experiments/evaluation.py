"""Experiment-only scoring and validation helpers for benchmark runs.

Evaluator labels belong outside the input workspace. These helpers are called
after predictions exist, or by the operator who records an external benchmark
run. A generated positive label is not the same as an expert judgment on a
real-world alert.
"""

from __future__ import annotations

import json
import hashlib
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from reflex.core.models.lenses import Observation


def score_synthetic_run(observations: list[Observation], truth: list[dict[str, str]], event_times: dict[str, datetime]) -> dict[str, Any]:
    """Score flagged observations against synthetic event-level labels.

    An observation is a true positive when it covers at least one labeled
    event for the same subject in its half-open interval. Recovered events are
    deduplicated by event ID, so an event covered by multiple observations is
    counted once in recall. Observation precision-style metrics and event-level
    recall therefore use different units.
    """
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
    """Return a deterministic percentile bootstrap interval for the sample mean.

    The fixed resampling seed makes reports reproducible. The interval
    summarizes the supplied sample; it does not correct for dataset or
    benchmark-design bias.
    """
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


def protocol_manifest(*, seeds: list[int], objective: str, inference_budget: int,
                      compute_budget_seconds: int) -> dict[str, Any]:
    """Create an unbound A–E protocol; runs cannot be recorded until inputs are pinned."""
    if not seeds or any(not _nonnegative_int(seed) for seed in seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be a nonempty list of distinct nonnegative integers")
    if not isinstance(objective, str) or not objective.strip():
        raise ValueError("objective must be nonempty text")
    if not _nonnegative_int(inference_budget) or not _nonnegative_int(compute_budget_seconds) or compute_budget_seconds == 0:
        raise ValueError("inference budget must be nonnegative and compute budget must be a positive integer")
    return {
        "protocol_version": 2,
        "objective": objective,
        "seeds": seeds,
        "batches": ["one-shot", "batch-1", "batch-2", "batch-3", "batch-4", "batch-5", "batch-6"],
        "equal_budgets": {"inference_tokens": inference_budget, "non_inference_compute_seconds": compute_budget_seconds},
        "shared_inputs": {"event_batches": "same bytes and order for all variants",
                          "ground_truth": "held by evaluator; never in agent workspace",
                          "statistical_artifacts": "same train cutoff, scaler, model weights, features and calibration threshold for C and D",
                          "expected_sha256": {}, "bound": False},
        "variants": [{"id": key, "description": value} for key, value in VARIANTS.items()],
        "run_record_required": ["variant", "seed", "batch", "input_hash", "objective", "output",
                                "inference_tokens", "inference_cost", "compute_seconds",
                                "investigation_count", "evidence_refs", "workspace_ref", "execution"],
        "records": [],
        "note": "No experiment result is implied by this manifest. Bind exact input files before recording runs.",
    }


def bind_protocol_input_hashes(manifest_path: str | Path,
                               input_hashes: dict[str, dict[str, str]]) -> None:
    """Pin one SHA-256 digest for every seed/batch cell before any run is recorded."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    if manifest.get("protocol_version") != 2:
        raise ValueError("input binding requires a protocol version 2 manifest")
    if manifest["records"]:
        raise ValueError("inputs must be bound before recording any runs")
    expected_seeds = {str(seed) for seed in manifest["seeds"]}
    if set(input_hashes) != expected_seeds:
        raise ValueError("input hashes must include exactly the protocol seeds")
    expected_batches = set(manifest["batches"])
    normalized: dict[str, dict[str, str]] = {}
    for seed in sorted(expected_seeds, key=int):
        batches = input_hashes[seed]
        if set(batches) != expected_batches:
            raise ValueError(f"seed {seed} must include exactly these batches: {', '.join(manifest['batches'])}")
        normalized[seed] = {}
        for batch in manifest["batches"]:
            digest = batches[batch]
            if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                raise ValueError(f"{seed}/{batch} must have a lowercase SHA-256 digest")
            normalized[seed][batch] = digest
    manifest["shared_inputs"]["expected_sha256"] = normalized
    manifest["shared_inputs"]["bound"] = True
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def append_run_record(manifest_path: str | Path, record: dict[str, Any]) -> None:
    """Validate a recorded run against pinned inputs, objective, budgets, and output artifact."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    if manifest.get("protocol_version") != 2:
        raise ValueError("run recording requires a protocol version 2 manifest")
    if not manifest["shared_inputs"].get("bound"):
        raise ValueError("bind exact input file hashes before recording runs")
    required = manifest["run_record_required"]
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError("run record missing required fields: " + ", ".join(missing))
    if not isinstance(record["variant"], str) or record["variant"] not in VARIANTS:
        raise ValueError("variant must be A, B, C, D, or E")
    if not _nonnegative_int(record["seed"]):
        raise ValueError("seed must be a nonnegative integer")
    if record["seed"] not in manifest["seeds"]:
        raise ValueError("seed is not part of the protocol")
    if not isinstance(record["batch"], str) or record["batch"] not in manifest["batches"]:
        raise ValueError("batch is not part of the protocol")
    if record["objective"] != manifest["objective"]:
        raise ValueError("run objective does not exactly match the preregistered objective")
    expected_hash = manifest["shared_inputs"]["expected_sha256"][str(record["seed"])][record["batch"]]
    if record["input_hash"] != expected_hash:
        raise ValueError("run input_hash does not match the preregistered batch bytes")
    if any((item["variant"], item["seed"], item["batch"]) == (record["variant"], record["seed"], record["batch"]) for item in manifest["records"]):
        raise ValueError("a run for this variant, seed, and batch is already recorded")
    if not _nonnegative_int(record["inference_tokens"]):
        raise ValueError("inference_tokens must be a nonnegative integer")
    if record["inference_tokens"] > manifest["equal_budgets"]["inference_tokens"]:
        raise ValueError("record exceeds the inference token budget")
    if not _nonnegative_number(record["compute_seconds"]):
        raise ValueError("compute_seconds must be a finite nonnegative number")
    if record["compute_seconds"] > manifest["equal_budgets"]["non_inference_compute_seconds"]:
        raise ValueError("record exceeds the compute budget")
    if not _nonnegative_number(record["inference_cost"]):
        raise ValueError("inference_cost must be a finite nonnegative number")
    if not _nonnegative_int(record["investigation_count"]):
        raise ValueError("investigation_count must be a nonnegative integer")
    if not isinstance(record["evidence_refs"], list) or any(not isinstance(item, str) or not item for item in record["evidence_refs"]):
        raise ValueError("evidence_refs must be a list of nonempty strings")
    if not isinstance(record["workspace_ref"], str) or not record["workspace_ref"].strip():
        raise ValueError("workspace_ref must identify the run workspace or commit")
    output = record["output"]
    if (not isinstance(output, dict) or not _nonnegative_int(output.get("validated_discoveries"))
            or not isinstance(output.get("hypotheses"), list)):
        raise ValueError("output must include validated_discoveries (integer) and hypotheses (list)")
    _verify_output_artifact(path, record["execution"], output)
    manifest["records"].append(record)
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def protocol_report(manifest_path: str | Path) -> dict[str, Any]:
    """Summarize records present in a protocol file and report completeness."""
    manifest = json.loads(Path(manifest_path).read_text())
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in manifest["records"]:
        grouped[record["variant"]].append(record)
    expected_cells = {(seed, batch) for seed in manifest["seeds"] for batch in manifest["batches"]}
    output = []
    for variant, description in VARIANTS.items():
        records = grouped[variant]
        actual_cells = {(record["seed"], record["batch"]) for record in records}
        missing = [{"seed": seed, "batch": batch} for seed, batch in sorted(expected_cells - actual_cells,
                                                                             key=lambda cell: (cell[0], cell[1]))]
        duplicates = max(0, len(records) - len(actual_cells))
        output.append({
            "variant": variant, "description": description, "recorded_runs": len(records),
            "expected_runs": len(expected_cells), "missing_runs": missing, "duplicate_runs": duplicates,
            "validated_discoveries": sum(r["output"]["validated_discoveries"] for r in records),
            "inference_tokens": sum(int(r["inference_tokens"]) for r in records),
            "inference_cost": sum(float(r["inference_cost"]) for r in records),
            "compute_seconds": sum(float(r["compute_seconds"]) for r in records),
            "investigations": sum(int(r["investigation_count"]) for r in records),
        })
    return {"protocol_version": manifest["protocol_version"], "objective": manifest["objective"],
            "input_hashes_bound": manifest["shared_inputs"].get("bound", False), "variants": output,
            "complete": bool(manifest["shared_inputs"].get("bound")) and all(
                not item["missing_runs"] and item["duplicate_runs"] == 0
                and item["recorded_runs"] == item["expected_runs"] for item in output),
            "recorded_runs": len(manifest["records"])}


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _nonnegative_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _verify_output_artifact(manifest_path: Path, execution: Any, expected_output: dict[str, Any]) -> None:
    if not isinstance(execution, dict):
        raise ValueError("execution must identify the system and saved output artifact")
    required = {"system", "version", "artifact_path", "artifact_sha256"}
    if not required.issubset(execution):
        raise ValueError("execution must include system, version, artifact_path, and artifact_sha256")
    for key in ("system", "version", "artifact_path"):
        if not isinstance(execution[key], str) or not execution[key].strip():
            raise ValueError(f"execution.{key} must be a nonempty string")
    if not re.fullmatch(r"[0-9a-f]{64}", str(execution["artifact_sha256"])):
        raise ValueError("execution.artifact_sha256 must be a lowercase SHA-256 digest")
    artifact_path = Path(execution["artifact_path"])
    if artifact_path.is_absolute():
        raise ValueError("execution.artifact_path must be relative to the protocol file")
    resolved = (manifest_path.parent / artifact_path).resolve()
    if manifest_path.parent.resolve() not in resolved.parents:
        raise ValueError("execution.artifact_path must stay inside the protocol directory")
    try:
        payload = resolved.read_bytes()
    except OSError as error:
        raise ValueError("execution output artifact is missing or unreadable") from error
    if hashlib.sha256(payload).hexdigest() != execution["artifact_sha256"]:
        raise ValueError("execution artifact hash does not match the saved file")
    try:
        artifact_output = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("execution output artifact must be JSON") from error
    if artifact_output != expected_output:
        raise ValueError("saved output artifact does not match the output in the run record")
