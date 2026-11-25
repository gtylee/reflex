"""Evaluate training-only REFLEX lens selection at a fixed review budget."""

from __future__ import annotations

from collections import Counter, defaultdict
import argparse
import csv
import json
from datetime import timedelta
from pathlib import Path
from statistics import mean
from typing import Any

from reflex.experiments.generators import default_configs, generate_scenario
from reflex.experiments.lens_selection import select_lens, simple_numeric_baseline
from reflex.experiments.metrics import mean_interval


REVIEW_BUDGET = 20
FIRST_SEED = 3001
SEED_COUNT = 30


def _top_ids(experiment, scores: dict[str, float], budget: int) -> tuple[str, ...]:
    test_events = [event for event in experiment.events if event.event_time >= experiment.calibration_end]
    ranked = sorted(test_events, key=lambda event: (-scores[event.event_id], event.event_id))
    return tuple(event.event_id for event in ranked[:budget])


def _metrics(predicted: tuple[str, ...], truth_ids: set[str]) -> dict[str, Any]:
    predicted_ids = set(predicted)
    true_positives = len(predicted_ids & truth_ids)
    false_positives = len(predicted_ids - truth_ids)
    false_negatives = len(truth_ids - predicted_ids)
    precision = true_positives / len(predicted_ids) if predicted_ids else 0.0
    recall = true_positives / len(truth_ids) if truth_ids else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "reviewed": len(predicted_ids),
    }


def _summarize(rows: list[dict[str, Any]], method: str, *, seed: int) -> dict[str, Any]:
    selected = [row for row in rows if row["method"] == method]
    tp = sum(row["true_positives"] for row in selected)
    fp = sum(row["false_positives"] for row in selected)
    fn = sum(row["false_negatives"] for row in selected)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    interval = mean_interval([row["f1"] for row in selected], seed=seed)
    return {
        "cases": len(selected),
        "reviewed_events": sum(row["reviewed"] for row in selected),
        "truth_events": sum(row["truth_count"] for row in selected),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "pooled_precision": precision,
        "pooled_recall": recall,
        "pooled_f1": f1,
        "mean_case_f1": mean(row["f1"] for row in selected) if selected else 0.0,
        "mean_case_f1_interval_95": interval,
    }


def _run(first_seed: int, seed_count: int, entities: int, days: int) -> dict[str, Any]:
    configs = default_configs(seed=first_seed, entities=entities, days=days)
    case_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []

    for config in configs:
        for seed in range(first_seed, first_seed + seed_count):
            seeded = config.model_copy(update={"seed": seed})
            dataset = generate_scenario(seeded)
            # Keep the calibration period before the planted midstream changes.
            # The generator's default calibration boundary is later and includes
            # some changes, which can teach the threshold to accept them.
            agent_input = dataset.agent_input().model_copy(update={
                "calibration_end": dataset.start + timedelta(days=days // 2),
            })

            # The selector receives only the visible input. Ground-truth labels
            # are accessed below, after both rankings and thresholds are frozen.
            selected = select_lens(agent_input)
            baseline = simple_numeric_baseline(agent_input)
            selected_ids = _top_ids(agent_input, selected.scores, REVIEW_BUDGET)
            baseline_ids = _top_ids(agent_input, baseline.scores, REVIEW_BUDGET)

            truth_ids = {label.event_id for label in dataset.ground_truth
                         if label.event_time >= agent_input.calibration_end}
            selected_metrics = _metrics(selected_ids, truth_ids)
            baseline_metrics = _metrics(baseline_ids, truth_ids)
            test_events = [event for event in agent_input.events
                           if event.event_time >= agent_input.calibration_end]
            selected_threshold_ids = tuple(
                event.event_id for event in test_events
                if (selected.scores[event.event_id] >= selected.threshold if selected.threshold > 0
                    else selected.scores[event.event_id] > 0)
            )
            baseline_threshold_ids = tuple(
                event.event_id for event in test_events
                if (baseline.scores[event.event_id] >= baseline.threshold if baseline.threshold > 0
                    else baseline.scores[event.event_id] > 0)
            )
            selected_threshold_metrics = _metrics(selected_threshold_ids, truth_ids)
            baseline_threshold_metrics = _metrics(baseline_threshold_ids, truth_ids)

            for method, lens, predictions, metrics, threshold_candidates in (
                ("selected", selected, selected_ids, selected_metrics, len(selected_threshold_ids)),
                ("numeric_baseline", baseline, baseline_ids, baseline_metrics, len(baseline_threshold_ids)),
            ):
                case_rows.append({
                    "scenario": config.scenario,
                    "seed": seed,
                    "method": method,
                    "lens_id": lens.lens_id,
                    "threshold": lens.threshold,
                    "threshold_test_candidates": threshold_candidates,
                    "truth_count": len(truth_ids),
                    **metrics,
                })
                for rank, event_id in enumerate(predictions, start=1):
                    prediction_rows.append({
                        "scenario": config.scenario,
                        "seed": seed,
                        "method": method,
                        "rank": rank,
                        "event_id": event_id,
                        "is_labeled": event_id in truth_ids,
                        "score": lens.scores[event_id],
                    })

            for method, metrics in (("selected_threshold", selected_threshold_metrics),
                                    ("numeric_baseline_threshold", baseline_threshold_metrics)):
                threshold_rows.append({
                    "scenario": config.scenario,
                    "seed": seed,
                    "method": method,
                    "truth_count": len(truth_ids),
                    **metrics,
                })
            for method, lens, predictions in (
                ("selected_threshold", selected, selected_threshold_ids),
                ("numeric_baseline_threshold", baseline, baseline_threshold_ids),
            ):
                for rank, event_id in enumerate(predictions, start=1):
                    prediction_rows.append({
                        "scenario": config.scenario,
                        "seed": seed,
                        "method": method,
                        "rank": rank,
                        "event_id": event_id,
                        "is_labeled": event_id in truth_ids,
                        "score": lens.scores[event_id],
                    })

            decisions.append({
                "scenario": config.scenario,
                "seed": seed,
                "selected_lens": selected.lens_id,
                "selected_features": list(selected.features),
                "rationale": selected.rationale,
                "training_profile": selected.training_profile,
                "calibration_threshold": selected.threshold,
                "calibration_events_above_threshold": selected.calibration_alert_count,
                "heldout_events_above_threshold": len(selected_threshold_ids),
                "review_budget": REVIEW_BUDGET,
                "selected_true_positives_at_budget": selected_metrics["true_positives"],
                "baseline_true_positives_at_budget": baseline_metrics["true_positives"],
                "selected_true_positives_at_threshold": selected_threshold_metrics["true_positives"],
                "baseline_true_positives_at_threshold": baseline_threshold_metrics["true_positives"],
            })

    summary = {
        method: _summarize(case_rows, method, seed=first_seed)
        for method in ("selected", "numeric_baseline")
    }
    by_scenario = {}
    for scenario in sorted({row["scenario"] for row in case_rows}):
        rows = [row for row in case_rows if row["scenario"] == scenario]
        by_scenario[scenario] = {
            method: _summarize(rows, method, seed=first_seed)
            for method in ("selected", "numeric_baseline")
        }
    threshold_summary = {
        method: _summarize(threshold_rows, method, seed=first_seed)
        for method in ("selected_threshold", "numeric_baseline_threshold")
    }
    threshold_by_scenario = {}
    for scenario in sorted({row["scenario"] for row in threshold_rows}):
        rows = [row for row in threshold_rows if row["scenario"] == scenario]
        threshold_by_scenario[scenario] = {
            method: _summarize(rows, method, seed=first_seed)
            for method in ("selected_threshold", "numeric_baseline_threshold")
        }
    return {
        "experiment": "reflex-training-only-lens-selection-v1",
        "protocol": {
            "first_seed": first_seed,
            "seed_count": seed_count,
            "seeds": list(range(first_seed, first_seed + seed_count)),
            "entities_per_case": entities,
            "days_per_case": days,
            "scenarios": [config.scenario for config in configs],
            "review_budget_per_case": REVIEW_BUDGET,
            "calibration_threshold_quantile": 0.99,
            "split": f"first {days // 3} days fit, days {days // 3} through {days // 2 - 1} calibrate, day {days // 2} onward held out",
            "selection_inputs": "training event types, numeric/categorical attributes, event density, and numeric correlations",
            "calibration_inputs": "calibration event scores only",
            "evaluation_unit": "exact labeled event ID after the calibration boundary",
            "labels_available_to_selector": False,
            "data_kind": "synthetic generated streams with held-out generator seeds",
        },
        "summary": summary,
        "by_scenario": by_scenario,
        "threshold_summary": threshold_summary,
        "threshold_by_scenario": threshold_by_scenario,
        "case_rows": case_rows,
        "threshold_rows": threshold_rows,
        "decisions": decisions,
        "predictions": prediction_rows,
        "limits": [
            "The data is synthetic and the generated mechanisms are known to the benchmark author.",
            "The selector uses transparent structural rules, not a trained meta-model.",
            "The 99th-percentile threshold is recorded for operational use. The equal-budget comparison ranks the same 20 held-out events per method.",
            "The simple baseline is the maximum robust numeric deviation across training fields.",
        ],
    }


def _write(output_dir: Path, report: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark-report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                       encoding="utf-8")
    for file_name, key in (("case-scores.csv", "case_rows"),
                           ("threshold-scores.csv", "threshold_rows"),
                           ("selection-decisions.csv", "decisions"),
                           ("reviewed-events.csv", "predictions")):
        rows = report[key]
        if rows:
            with (output_dir / file_name).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows({name: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                                  for name, value in row.items()} for row in rows)

    selected = report["summary"]["selected"]
    baseline = report["summary"]["numeric_baseline"]
    selected_threshold = report["threshold_summary"]["selected_threshold"]
    baseline_threshold = report["threshold_summary"]["numeric_baseline_threshold"]
    lines = [
        "# REFLEX training-only lens selection", "",
        f"- Synthetic scenarios: {len(report['protocol']['scenarios'])}.",
        f"- Seeds: {report['protocol']['first_seed']}–{report['protocol']['seeds'][-1]} ({report['protocol']['seed_count']} per scenario).",
        f"- Human review budget: {report['protocol']['review_budget_per_case']} events per case, matched for both methods.",
        f"- Split: {report['protocol']['split']}.",
        "- Lens choice uses training event structure and attributes. The threshold uses calibration scores. Evaluation labels are accessed only after predictions are frozen.",
        "", "## Equal review budget results", "",
        "| Method | Reviewed | Truth events | True positives | Precision | Recall | F1 | Mean case F1 (95% interval) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in (("Selected lens", selected), ("Numeric baseline", baseline)):
        interval = result["mean_case_f1_interval_95"]
        lines.append(
            f"| {name} | {result['reviewed_events']:,} | {result['truth_events']:,} | {result['true_positives']:,} "
            f"| {result['pooled_precision']:.3f} | {result['pooled_recall']:.3f} | {result['pooled_f1']:.3f} "
            f"| {result['mean_case_f1']:.3f} ({interval['lower']:.3f}–{interval['upper']:.3f}) |"
        )
    lines += ["", "## Frozen threshold results", "",
              "These counts apply each method's calibration threshold without the review cap, so they show its natural alert volume.",
              "", "| Method | Alerts | Truth events | True positives | Precision | Recall | F1 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, result in (("Selected lens", selected_threshold), ("Numeric baseline", baseline_threshold)):
        lines.append(
            f"| {name} | {result['reviewed_events']:,} | {result['truth_events']:,} | {result['true_positives']:,} "
            f"| {result['pooled_precision']:.3f} | {result['pooled_recall']:.3f} | {result['pooled_f1']:.3f} |"
        )
    lines += ["", "## Per-scenario F1", "",
              "| Scenario | Selected lens F1 | Numeric baseline F1 | Lens choices |",
              "|---|---:|---:|---|"]
    choices: dict[str, Counter] = defaultdict(Counter)
    for item in report["decisions"]:
        choices[item["scenario"]][item["selected_lens"]] += 1
    for scenario, methods in report["by_scenario"].items():
        lines.append(f"| {scenario} | {methods['selected']['pooled_f1']:.3f} "
                     f"| {methods['numeric_baseline']['pooled_f1']:.3f} "
                     f"| {', '.join(f'{lens} × {count}' for lens, count in sorted(choices[scenario].items()))} |")
    lines += ["", "## Limits", "", *[f"- {item}" for item in report["limits"]],
              "", "## Files", "",
              "- [`selection-decisions.csv`](selection-decisions.csv): lens, rationale, training profile, and frozen threshold for each case.",
              "- [`case-scores.csv`](case-scores.csv): metrics for each method and seed.",
              "- [`threshold-scores.csv`](threshold-scores.csv): performance at each frozen threshold, without the review cap.",
              "- [`reviewed-events.csv`](reviewed-events.csv): the exact 20 events each method sent to review.",
              "- [`benchmark-report.json`](benchmark-report.json): complete protocol, per-scenario results, and limitations."]
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-seed", type=int, default=FIRST_SEED)
    parser.add_argument("--seed-count", type=int, default=SEED_COUNT)
    parser.add_argument("--entities", type=int, default=40)
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--out", type=Path,
                        default=Path("reports/reflex-experiments/lens-selection-milestone"))
    args = parser.parse_args()
    if args.first_seed < 0 or args.seed_count < 2 or args.entities < 4 or args.days < 10:
        parser.error("first-seed must be nonnegative, seed-count at least 2, entities at least 4, and days at least 10")
    report = _run(args.first_seed, args.seed_count, args.entities, args.days)
    _write(args.out, report)
    print(json.dumps({
        "report": str((args.out / "README.md").resolve()),
        "selection_summary": report["summary"],
        "threshold_summary": report["threshold_summary"],
        "by_scenario": report["by_scenario"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
