"""Fair A/B/C benchmark manifests and recording; baselines are never fabricated."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .news_demo import benchmark_input_hash, run_news_demo


METRIC_NAMES = (
    "alert_precision", "alert_recall", "duplicate_suppression", "decision_consistency",
    "latency_ms", "inference_tokens", "compute_cost_seconds", "evidence_completeness",
    "replay_reproducibility", "action_reliability",
)
VARIANTS = ("A", "B", "C")


def benchmark_manifest(*, seeds: list[int] | None = None, objective: str | None = None,
                       inference_token_budget: int = 20000, compute_budget_seconds: int = 900) -> dict[str, Any]:
    if inference_token_budget < 0 or compute_budget_seconds <= 0:
        raise ValueError("benchmark budgets must be non-negative tokens and positive seconds")
    return {
        "protocol": "reflex-news-ab-c", "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_hash": benchmark_input_hash(),
        "input_bundle": "synthetic portfolio holdings, five news events, and one manager response",
        "objective": objective or "Identify portfolio-relevant developments, cite event evidence, suppress duplicates, and state uncertainty.",
        "variants": {"A": "direct LLM processing", "B": "tool-enabled agent processing", "C": "REFLEX Reaction processing"},
        "shared_conditions": {
            "same_input_hash_required": True,
            "same_portfolio_state": True,
            "same_event_history": True,
            "same_information_availability": True,
            "same_objective_and_evaluation_labels": True,
            "actions": "Record intended notifications; external delivery is disabled for all benchmark variants.",
        },
        "budgets": {"inference_tokens": inference_token_budget, "compute_seconds": compute_budget_seconds},
        "seeds": seeds or [11, 23, 37],
        "metrics": list(METRIC_NAMES),
        "runs": [],
        "baseline_status": {"A": "not recorded", "B": "not recorded", "C": "not recorded"},
    }


def append_run(protocol_path: str | Path, run: dict[str, Any]) -> dict[str, Any]:
    path = Path(protocol_path)
    manifest = json.loads(path.read_text())
    _validate_run(run, manifest)
    duplicate = any(item["variant"] == run["variant"] and item["seed"] == run["seed"] for item in manifest["runs"])
    if duplicate:
        raise ValueError(f"variant {run['variant']} seed {run['seed']} is already recorded")
    stored = {**run, "recorded_at": datetime.now(timezone.utc).isoformat()}
    manifest["runs"].append(stored)
    expected = set(manifest["seeds"])
    for variant in VARIANTS:
        present = {item["seed"] for item in manifest["runs"] if item["variant"] == variant}
        manifest["baseline_status"][variant] = (
            "recorded" if expected.issubset(present) else "partially recorded" if present else "not recorded"
        )
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def report(protocol_path: str | Path) -> dict[str, Any]:
    manifest = json.loads(Path(protocol_path).read_text())
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in manifest["runs"]:
        grouped[run["variant"]].append(run)
    expected_seeds = set(manifest["seeds"])
    variant_summaries = {}
    for variant in VARIANTS:
        records = grouped.get(variant, [])
        present_seeds = {run["seed"] for run in records}
        missing_seeds = sorted(expected_seeds - present_seeds)
        variant_summaries[variant] = {
            "runs": len(records), "missing_seeds": missing_seeds,
            "mean_metrics": {metric: sum(float(row["metrics"][metric]) for row in records) / len(records)
                             for metric in METRIC_NAMES} if records else None,
        }
    return {
        "protocol": manifest["protocol"], "input_hash": manifest["input_hash"],
        "recorded_runs": len(manifest["runs"]), "baseline_status": manifest["baseline_status"],
        "variants": variant_summaries,
        "comparisons_ready": all(not summary["missing_seeds"] for summary in variant_summaries.values()),
    }


def run_reflex_variant(protocol_path: str | Path, out_dir: str | Path, *, seed: int = 11) -> dict[str, Any]:
    manifest = json.loads(Path(protocol_path).read_text())
    if manifest["input_hash"] != benchmark_input_hash():
        raise ValueError("the current REFLEX case differs from the manifest input hash")
    if seed not in manifest["seeds"]:
        raise ValueError(f"seed {seed} is not declared by the benchmark manifest")
    demo = run_news_demo(out_dir)
    actions = demo.get("action_events", [])
    actual = {(item["story_id"], item["revision"]) for item in actions}
    expected = {("news-100", 0), ("news-100", 1)}
    true_positives = len(actual & expected)
    precision = true_positives / len(actual) if actual else 1.0
    recall = true_positives / len(expected)
    successful_ids = {item["action_event_id"] for item in demo.get("action_results", []) if item["status"] == "succeeded"}
    action_ids = {item["event_id"] for item in actions}
    metrics = {
        "alert_precision": precision, "alert_recall": recall,
        "duplicate_suppression": float(demo["scenarios"]["duplicate_suppressed"]),
        "decision_consistency": float(demo["scenarios"]["replay_decisions_match"]),
        "latency_ms": float(demo["wall_seconds"] * 1000), "inference_tokens": 0.0,
        "compute_cost_seconds": float(demo["wall_seconds"]),
        "evidence_completeness": sum(bool(item["parent_event_ids"]) for item in actions) / max(len(actions), 1),
        "replay_reproducibility": float(demo["scenarios"]["replay_decisions_match"]),
        "action_reliability": len(action_ids & successful_ids) / max(len(action_ids), 1),
    }
    run = {
        "run_id": str(uuid4()), "variant": "C", "seed": seed,
        "input_hash": demo["input_hash"], "objective": manifest["objective"],
        "metrics": metrics,
        "resource_usage": {"inference_tokens": 0, "compute_seconds": demo["wall_seconds"]},
        "evidence_refs": [item["parent_event_ids"][0] for item in actions if item["parent_event_ids"]],
        "output": {"action_event_ids": [item["event_id"] for item in actions],
                   "scenario_results": demo["scenarios"]},
        "run_ref": str(Path(out_dir) / "portfolio-news-demo.json"),
    }
    append_run(protocol_path, run)
    return run


def _validate_run(run: dict[str, Any], manifest: dict[str, Any]) -> None:
    required = {"run_id", "variant", "seed", "input_hash", "objective", "metrics", "resource_usage", "evidence_refs", "output", "run_ref"}
    missing = required - set(run)
    if missing:
        raise ValueError(f"run is missing required field(s): {', '.join(sorted(missing))}")
    if run["variant"] not in VARIANTS:
        raise ValueError("variant must be A, B, or C")
    if run["seed"] not in manifest["seeds"]:
        raise ValueError("run seed is not declared by the protocol")
    if run["input_hash"] != manifest["input_hash"]:
        raise ValueError("run input_hash does not match the shared benchmark bundle")
    if run["objective"] != manifest["objective"]:
        raise ValueError("run objective does not match the shared protocol")
    if set(run["metrics"]) != set(METRIC_NAMES):
        raise ValueError("run must provide every declared metric and no undeclared metrics")
    for metric, value in run["metrics"].items():
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"metric {metric!r} must be a finite non-negative number")
        if metric not in {"latency_ms", "inference_tokens", "compute_cost_seconds"} and value > 1:
            raise ValueError(f"metric {metric!r} must be between 0 and 1")
    usage = run["resource_usage"]
    if not isinstance(usage, dict) or not {"inference_tokens", "compute_seconds"}.issubset(usage):
        raise ValueError("resource_usage must include inference_tokens and compute_seconds")
    for key in ("inference_tokens", "compute_seconds"):
        if not isinstance(usage[key], (int, float)) or not math.isfinite(usage[key]) or usage[key] < 0:
            raise ValueError(f"resource_usage.{key} must be finite and non-negative")
    if usage["inference_tokens"] > manifest["budgets"]["inference_tokens"]:
        raise ValueError("run exceeds the inference token budget")
    if usage["compute_seconds"] > manifest["budgets"]["compute_seconds"]:
        raise ValueError("run exceeds the compute time budget")
    if float(run["metrics"]["inference_tokens"]) != float(usage["inference_tokens"]):
        raise ValueError("inference_tokens metric must match resource_usage")
    if float(run["metrics"]["compute_cost_seconds"]) != float(usage["compute_seconds"]):
        raise ValueError("compute_cost_seconds metric must match resource_usage")
    if not isinstance(run["evidence_refs"], list) or not isinstance(run["output"], dict) or not isinstance(run["run_ref"], str):
        raise ValueError("evidence_refs, output, and run_ref must be structured run fields")


__all__ = ["benchmark_manifest", "append_run", "report", "run_reflex_variant", "METRIC_NAMES"]
