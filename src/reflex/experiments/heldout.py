"""Repeatable multi-seed evaluation on unseen synthetic generator seeds."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from .generators import SCENARIOS, default_configs, generate_scenario
from .metrics import mean_interval
from .models.results import ExperimentReport
from .runner import run_comparison


AGGREGATE_METRICS = (
    "precision", "recall", "f1", "mechanism_recall", "evidence_completeness",
    "evidence_integrity", "evidence_temporal_validity", "evidence_entity_alignment",
    "evidence_target_inclusion", "evidence_supported_prediction_rate",
)


def run_heldout_benchmark(*, first_seed: int = 101, seed_count: int = 30,
                          entities: int = 40, days: int = 60,
                          scenarios: Iterable[str] | None = None) -> dict[str, Any]:
    """Run the deterministic C adapter over new seeds and exact event labels."""
    if first_seed < 0 or seed_count < 2 or entities < 4 or days < 10:
        raise ValueError("first_seed must be nonnegative, seed_count at least 2, entities at least 4, and days at least 10")
    selected_scenarios = tuple(SCENARIOS if scenarios is None else scenarios)
    if not selected_scenarios or len(selected_scenarios) != len(set(selected_scenarios)):
        raise ValueError("select one or more unique scenarios")
    unknown = set(selected_scenarios) - set(SCENARIOS)
    if unknown:
        raise ValueError("unknown scenario(s): " + ", ".join(sorted(unknown)))

    seeds = list(range(first_seed, first_seed + seed_count))
    scenario_results: dict[str, Any] = {}
    for scenario in selected_scenarios:
        seed_runs: list[dict[str, Any]] = []
        completed_metrics: list[dict[str, Any]] = []
        for seed in seeds:
            config = next(item for item in default_configs(seed=seed, entities=entities, days=days)
                          if item.scenario == scenario)
            dataset = generate_scenario(config)
            report: ExperimentReport = run_comparison(config, dataset, variants=("C",))
            run = report.runs[0]
            metrics = run.metrics if run.status == "completed" else None
            if metrics is not None:
                completed_metrics.append(metrics)
            seed_runs.append({
                "seed": seed,
                "input_hash": report.input_hash,
                "status": run.status,
                "execution_kind": run.execution_kind,
                "detector": run.output.metadata.get("detector") if run.output else None,
                "generated_event_count": report.generated_event_count,
                "test_truth_event_count": report.test_truth_event_count,
                "predicted_event_count": metrics["predicted_events"] if metrics else None,
                "metrics": metrics,
                "error": run.error,
            })

        metric_summary: dict[str, Any] = {}
        for metric in AGGREGATE_METRICS:
            values = [float(item[metric]) for item in completed_metrics if item.get(metric) is not None]
            interval = mean_interval(values, seed=first_seed)
            metric_summary[metric] = {**interval,
                                      "median": median(values) if values else 0.0,
                                      "min": min(values) if values else 0.0,
                                      "max": max(values) if values else 0.0}

        tp = sum(int(item["true_positives"]) for item in completed_metrics)
        fp = sum(int(item["false_positives"]) for item in completed_metrics)
        fn = sum(int(item["false_negatives"]) for item in completed_metrics)
        pooled_precision = tp / (tp + fp) if tp + fp else 0.0
        pooled_recall = tp / (tp + fn) if tp + fn else 0.0
        scenario_results[scenario] = {
            "requested_seed_count": len(seeds),
            "completed_seed_count": len(completed_metrics),
            "heldout_test_truth_events": sum(item["test_truth_event_count"] for item in seed_runs),
            "pooled_event_metrics": {
                "true_positives": tp, "false_positives": fp, "false_negatives": fn,
                "precision": pooled_precision, "recall": pooled_recall,
                "f1": (2 * pooled_precision * pooled_recall / (pooled_precision + pooled_recall)
                      if pooled_precision + pooled_recall else 0.0),
            },
            "per_seed_metric_summary": metric_summary,
            "runs": seed_runs,
        }

    return {
        "experiment": "reflex-heldout-synthetic-seeds-v1",
        "protocol": {
            "first_seed": first_seed,
            "seed_count": seed_count,
            "seeds": seeds,
            "entities_per_case": entities,
            "days_per_case": days,
            "scenarios": list(selected_scenarios),
            "variant": "C",
            "evaluation_unit": "exact event ID after the calibration boundary",
            "uncertainty_unit": "seed; 95% bootstrap interval across held-out seeds",
            "evidence_integrity_rule": "Each reference must exist in that case, share an entity with its prediction, and be no later than the predicted event. This checks structural support, not causal sufficiency.",
            "label_source": "held-out synthetic generator ground truth",
            "expert_labeled": False,
            "comparative_claim_allowed": False,
        },
        "scenarios": scenario_results,
        "limitations": [
            "Seeds 101 onward are unseen generator seeds, not independently collected real-world cases.",
            "Labels are generated from known synthetic mechanisms and are not expert judgments.",
            "Only deterministic variant C is available in this run; it does not compare REFLEX with an agent baseline.",
            "Evidence integrity is an entity/time consistency check and does not establish that evidence is causally sufficient.",
        ],
    }


def write_heldout_report(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "heldout-report.json"
    markdown_path = root / "heldout-report.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    markdown_path.write_text(_markdown_report(report))
    return json_path, markdown_path


def _markdown_report(report: dict[str, Any]) -> str:
    protocol = report["protocol"]
    lines = [
        "# REFLEX held-out synthetic benchmark",
        "",
        f"Seeds: {protocol['first_seed']}–{protocol['seeds'][-1]} ({protocol['seed_count']} per scenario).",
        f"Cases: {protocol['entities_per_case']} entities × {protocol['days_per_case']} days. Variant: {protocol['variant']}.",
        "Labels are held-out synthetic generator ground truth, not expert-labeled real cases.",
        "Predictions are scored by exact event ID after calibration. Intervals resample seeds.",
        "",
        "| Scenario | Seeds complete | Truth events | Precision | Recall | F1 | Evidence integrity |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for scenario, result in report["scenarios"].items():
        metrics = result["per_seed_metric_summary"]
        lines.append(
            f"| {scenario} | {result['completed_seed_count']}/{result['requested_seed_count']} "
            f"| {result['heldout_test_truth_events']} "
            f"| {_format_interval(metrics['precision'])} "
            f"| {_format_interval(metrics['recall'])} "
            f"| {_format_interval(metrics['f1'])} "
            f"| {_format_interval(metrics['evidence_integrity'])} |"
        )
    lines.extend(["", "## Interpretation", "", *[f"- {item}" for item in report["limitations"]], ""])
    return "\n".join(lines)


def _format_interval(summary: dict[str, Any]) -> str:
    return f"{summary['mean']:.3f} ({summary['lower']:.3f}–{summary['upper']:.3f})"


__all__ = ["run_heldout_benchmark", "write_heldout_report"]
