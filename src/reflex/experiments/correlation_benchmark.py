"""Paired, label-blind lens comparison for a delayed correlation shift.

This profile is intentionally separate from the legacy phase-one result. It
keeps training and calibration in the unchanged regime, selects thresholds
without labels, then scores each candidate on the same held-out events.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest

from .adapters import QuantileLensAdapter, default_phase_one_adapters, feature_value
from .generators import default_configs, generate_scenario
from .metrics import mean_interval, score_predictions
from .models.inputs import RunBudget


METHODS = (
    "legacy_product_quantile",
    "calibrated_product_quantile",
    "calibrated_rolling_correlation",
    "calibrated_mahalanobis",
    "calibrated_isolation_forest",
)
METRICS = ("precision", "recall", "f1", "predicted_events", "false_positives",
           "stable_false_positive_rate")
WINDOW = 24
CALIBRATION_QUANTILE = 0.99


def _rows(dataset):
    ordered = sorted(dataset.events, key=lambda event: (event.event_time, event.event_id))
    train = [event for event in ordered if event.event_time < dataset.train_end]
    calibration = [event for event in ordered
                   if dataset.train_end <= event.event_time < dataset.calibration_end]
    test = [event for event in ordered if event.event_time >= dataset.calibration_end]
    return ordered, train, calibration, test


def _mahalanobis_scores(events, train):
    train_x = np.asarray([[feature_value(event, name) for name in ("factor_x", "factor_y")]
                          for event in train], dtype=float)
    x = np.asarray([[feature_value(event, name) for name in ("factor_x", "factor_y")]
                    for event in events], dtype=float)
    mean = train_x.mean(axis=0)
    covariance = np.cov(train_x, rowvar=False)
    covariance += np.eye(covariance.shape[0]) * 1e-6
    inverse = np.linalg.pinv(covariance)
    centered = x - mean
    return np.einsum("ij,jk,ik->i", centered, inverse, centered)


def _run_candidates(dataset, seed: int) -> tuple[dict[str, Any], dict[str, float]]:
    ordered, train, calibration, test = _rows(dataset)
    test_ids = {event.event_id for event in test}
    budget = RunBudget()
    legacy = QuantileLensAdapter("product_xy", direction="lower", quantile=0.995).run(
        dataset.agent_input(), budget, trial_seed=seed)

    product_cal = np.asarray([feature_value(event, "product_xy") for event in calibration])
    product_threshold = float(np.quantile(product_cal, 0.01))
    product_predictions = {event.event_id for event in test
                           if feature_value(event, "product_xy") < product_threshold}

    corr_adapter = default_phase_one_adapters(dataset.config)["C"]
    corr_output = corr_adapter.run(dataset.agent_input(), budget, trial_seed=seed)
    corr_predictions = set(corr_output.predicted_event_ids) & test_ids
    corr_evidence = {event_id: corr_output.evidence_refs[event_id] for event_id in corr_predictions}

    all_mahal = _mahalanobis_scores(ordered, train)
    score_by_id = {event.event_id: float(score) for event, score in zip(ordered, all_mahal, strict=True)}
    mahal_threshold = float(np.quantile([score_by_id[event.event_id] for event in calibration],
                                        CALIBRATION_QUANTILE))
    mahal_predictions = {event.event_id for event in test
                         if score_by_id[event.event_id] > mahal_threshold}

    train_xy = np.asarray([[feature_value(event, name) for name in ("factor_x", "factor_y")]
                           for event in train], dtype=float)
    cal_xy = np.asarray([[feature_value(event, name) for name in ("factor_x", "factor_y")]
                         for event in calibration], dtype=float)
    test_xy = np.asarray([[feature_value(event, name) for name in ("factor_x", "factor_y")]
                          for event in test], dtype=float)
    forest = IsolationForest(n_estimators=200, contamination="auto", random_state=seed, n_jobs=1)
    forest.fit(train_xy)
    cal_scores = forest.score_samples(cal_xy)
    forest_threshold = float(np.quantile(cal_scores, 0.01))
    forest_predictions = {event.event_id for event, score in zip(test, forest.score_samples(test_xy), strict=True)
                          if score < forest_threshold}

    outputs = {
        "legacy_product_quantile": (set(legacy.predicted_event_ids) & test_ids,
                                    legacy.evidence_refs, {"threshold": legacy.metadata.get("threshold"),
                                                           "threshold_source": "training 0.5th percentile"}),
        "calibrated_product_quantile": (product_predictions, {},
                                        {"threshold": product_threshold,
                                         "threshold_source": "calibration 1st percentile"}),
        "calibrated_rolling_correlation": (corr_predictions, corr_evidence,
                                            {"threshold": corr_output.metadata.get("threshold"),
                                             "threshold_source": "calibration 99th percentile",
                                             "training_correlation": corr_output.metadata.get("baseline_correlation"),
                                             "window_events": WINDOW}),
        "calibrated_mahalanobis": (mahal_predictions, {},
                                   {"threshold": mahal_threshold,
                                    "threshold_source": "calibration 99th percentile"}),
        "calibrated_isolation_forest": (forest_predictions, {},
                                        {"threshold": forest_threshold,
                                         "threshold_source": "calibration 1st percentile"}),
    }
    truth = list(dataset.ground_truth)
    universe = [event.event_id for event in test]
    shift_day = int(dataset.config.parameters["correlation_shift_day"])
    stable_test_ids = {event.event_id for event in test
                       if (event.event_time - dataset.start).days < shift_day}
    results = {}
    for method, (predictions, evidence, diagnostics) in outputs.items():
        metrics = score_predictions(predictions, truth, evidence, universe_event_ids=universe,
                                    bootstrap_resamples=0)
        stable_false_positives = len(set(predictions) & stable_test_ids)
        results[method] = {**metrics, **diagnostics,
                           "stable_false_positives": stable_false_positives,
                           "stable_false_positive_rate": (stable_false_positives / len(stable_test_ids)
                                                          if stable_test_ids else 0.0)}
    return results, {"train": len(train), "calibration": len(calibration), "test": len(test)}


def run_correlation_benchmark(*, first_seed: int = 101, seed_count: int = 30,
                              entities: int = 40, days: int = 60) -> dict[str, Any]:
    """Run a paired multi-seed comparison with a clean pre-change calibration set."""
    if first_seed < 0 or seed_count < 2 or entities < 4 or days < 18:
        raise ValueError("first_seed must be nonnegative, seed_count at least 2, entities at least 4, and days at least 18")
    seeds = list(range(first_seed, first_seed + seed_count))
    runs = []
    for seed in seeds:
        base = next(config for config in default_configs(seed=seed, entities=entities, days=days)
                    if config.scenario == "correlation_shift")
        calibration_day = (days * 2) // 3
        test_days = days - calibration_day
        shift_day = calibration_day + test_days // 2
        config = base.model_copy(update={
            "version": "2.0",
            "parameters": {**base.parameters,
                           "generator_version": "correlation-shift-v2",
                           "lens_strategy": "calibrated_rolling_correlation",
                           "correlation_shift_day": shift_day,
                           "correlation_label_start_day": shift_day,
                           "correlation_label_end_day": days},
        })
        dataset = generate_scenario(config)
        candidate_results, split_counts = _run_candidates(dataset, seed)
        train_truth = sum(label.event_time < dataset.train_end for label in dataset.ground_truth)
        calibration_truth = sum(dataset.train_end <= label.event_time < dataset.calibration_end
                                for label in dataset.ground_truth)
        stable_test_ids = {event.event_id for event in dataset.events
                           if event.event_time >= dataset.calibration_end
                           and (event.event_time - dataset.start).days < shift_day}
        stable_truth = sum(label.event_id in stable_test_ids
                           for label in dataset.ground_truth)
        runs.append({"seed": seed, "input_hash": dataset.input_hash, "truth_event_count": len(dataset.ground_truth),
                     "training_truth_count": train_truth, "calibration_truth_count": calibration_truth,
                     "stable_test_truth_count": stable_truth,
                     "shift_day": shift_day, "split_counts": split_counts, "methods": candidate_results})

    for run in runs:
        if len({run["input_hash"] for _method in METHODS}) != 1:
            raise AssertionError("paired candidates did not share an input")
    summaries: dict[str, Any] = {}
    baseline = "legacy_product_quantile"
    for method in METHODS:
        summaries[method] = {}
        for metric in METRICS:
            values = [float(run["methods"][method][metric]) for run in runs]
            summary = mean_interval(values, seed=first_seed)
            summaries[method][metric] = {**summary, "median": median(values), "min": min(values), "max": max(values)}
        paired = [float(run["methods"][method]["f1"]) - float(run["methods"][baseline]["f1"])
                  for run in runs]
        summaries[method]["f1_delta_vs_legacy"] = mean_interval(paired, seed=first_seed)

    report = {
        "experiment": "reflex-correlation-shift-paired-v2",
        "protocol": {
            "seeds": seeds, "seed_count": seed_count, "entities_per_case": entities,
            "days_per_case": days, "paired_inputs": True,
            "input_hashes": {str(run["seed"]): run["input_hash"] for run in runs},
            "split": "first third training; second third calibration; final third held out",
            "shift_timing": "shift begins halfway through held-out period; training and calibration remain unchanged",
            "generator": "factor marginals stay standard-normal while correlation changes from +0.82 to -0.82",
            "selected_lens": "calibrated_rolling_correlation",
            "evidence_scope": "The lens cites a cross-entity factor window. This report scores event detection, not the framework's same-entity evidence-integrity metric.",
            "threshold_quantile": CALIBRATION_QUANTILE,
            "label_policy": "all held-out events from the change point onward are labeled as belonging to the changed regime",
            "evaluation_unit": "exact event IDs in the held-out period",
            "selection_rule": "candidate families and thresholds were specified before held-out scoring; only training and calibration data determine fitted values and thresholds",
            "label_use_for_selection": False,
            "external_validation": "See the existing NAB real-series reports; NAB labels are public window annotations, not independent expert adjudication, and the task differs from correlation-regime detection.",
        },
        "methods": summaries,
        "runs": runs,
        "limitations": [
            "This is a controlled synthetic test, not evidence on FX transactions or real market regimes.",
            "The benchmark evaluates known candidate lenses; it does not prove that an automated lens selector will choose correctly on unseen domains.",
            "The shifted regime occupies only half of each held-out period; the unchanged half provides a held-out false-positive check.",
            "The external NAB results use independently published anomaly windows, but no blind expert adjudication was performed and the data are univariate time series.",
            "Correlation evidence is cohort-wide. The current entity-local evidence-integrity check is not a suitable quality measure for these references and is not reported here.",
            "A/B/D/E agent runs remain unavailable; this experiment does not establish REFLEX superiority over a general-purpose agent.",
        ],
    }
    return report


def write_correlation_benchmark(report: dict[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    """Write a human-readable report, full JSON, paired-seed table, sample, and chart."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "correlation-shift-report.json"
    markdown_path = root / "README.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n")
    with (root / "paired-seed-results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("seed", "input_hash", "method", "precision", "recall", "f1",
                                                     "predicted_events", "false_positives", "stable_false_positives",
                                                     "stable_false_positive_rate", "threshold"))
        writer.writeheader()
        for run in report["runs"]:
            for method, result in run["methods"].items():
                writer.writerow({"seed": run["seed"], "input_hash": run["input_hash"], "method": method,
                                 **{key: result[key] for key in ("precision", "recall", "f1", "predicted_events",
                                                                  "false_positives", "stable_false_positives",
                                                                  "stable_false_positive_rate", "threshold")}})

    sample_run = report["runs"][0]
    config = next(item for item in default_configs(seed=sample_run["seed"],
                                                    entities=report["protocol"]["entities_per_case"],
                                                    days=report["protocol"]["days_per_case"])
                  if item.scenario == "correlation_shift")
    calibration_day = (report["protocol"]["days_per_case"] * 2) // 3
    shift_day = sample_run["shift_day"]
    config = config.model_copy(update={"version": "2.0", "parameters": {**config.parameters,
        "generator_version": "correlation-shift-v2", "correlation_shift_day": shift_day,
        "lens_strategy": "calibrated_rolling_correlation",
        "correlation_label_start_day": shift_day,
        "correlation_label_end_day": report["protocol"]["days_per_case"]}})
    sample_dataset = generate_scenario(config)
    event_labels = {label.event_id for label in sample_dataset.ground_truth}
    with (root / "sample-data.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("event_id", "event_time", "split", "factor_x", "factor_y", "product_xy", "evaluation_label"))
        selected = []
        for split, predicate in (
            ("training", lambda day: day < report["protocol"]["days_per_case"] // 3),
            ("calibration", lambda day: report["protocol"]["days_per_case"] // 3 <= day < calibration_day),
            ("heldout_stable", lambda day: calibration_day <= day < shift_day),
            ("heldout_shifted", lambda day: day >= shift_day),
        ):
            candidates = [event for event in sample_dataset.events
                          if predicate((event.event_time - sample_dataset.start).days)]
            selected.extend((split, event) for event in candidates[:3])
        for split, event in selected:
            attrs = event.attributes
            writer.writerow((event.event_id, event.event_time.isoformat(), split, attrs["factor_x"], attrs["factor_y"],
                             attrs["product_xy"], "changed_regime" if event.event_id in event_labels else "normal"))

    summaries = report["methods"]
    lines = ["# Paired correlation-shift benchmark", "",
             f"{report['protocol']['seed_count']} paired seeds ({report['protocol']['seeds'][0]}–{report['protocol']['seeds'][-1]}), "
             f"{report['protocol']['entities_per_case']} entities, {report['protocol']['days_per_case']} days.",
             "The baseline regime is used for training and calibration. The change begins halfway through the held-out period, leaving a stable held-out segment to measure false positives.",
             "Thresholds and fitted values are determined without using evaluation labels.",
             "Intervals are 95% percentile bootstrap intervals across seed-level results; F1 changes are paired by seed against the legacy lens.", "",
             "| Lens | Precision | Recall | F1 | Stable held-out FPR | F1 change vs legacy |",
             "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        result = summaries[method]
        lines.append(f"| {method} | {_fmt(result['precision'])} | {_fmt(result['recall'])} | {_fmt(result['f1'])} | "
                     f"{_fmt(result['stable_false_positive_rate'])} | "
                     f"{_fmt(result['f1_delta_vs_legacy'], interval=True)} |")
    lines.extend(["", "## Lens and threshold choices", "",
                  "- `legacy_product_quantile`: the original lower-tail `product_xy` quantile lens; cutoff fitted on training data.",
                  "- `calibrated_product_quantile`: same univariate feature, with a lower-tail cutoff from calibration data.",
                  "- `calibrated_rolling_correlation`: rolling 24-event correlation distance from the training correlation; threshold is the calibration 99th percentile.",
                  "- `calibrated_mahalanobis`: joint factor distance using training mean/covariance; threshold is the calibration 99th percentile.",
                  "- `calibrated_isolation_forest`: joint factor anomaly score fitted on training data; threshold is the calibration 1st score percentile.",
                  "", "## Reading the result", "",
                  "Each method sees identical event IDs for a seed, and each row in `paired-seed-results.csv` includes the shared input hash. F1 changes are paired by seed against the legacy lens.",
             "The rolling-correlation lens cites a cross-entity window. This report evaluates detection only; the current same-entity evidence-integrity check does not assess cohort evidence correctly.",
                  "The generated labels mark the changed regime, so this measures detection of a known synthetic distribution change. It does not show that an alert is a real-world anomaly.",
                  "", "## Files", "", "- `correlation-shift-report.json`: full protocol, diagnostics, hashes, and per-seed results.",
                  "- `paired-seed-results.csv`: per-seed paired candidate metrics and thresholds.",
                  "- `sample-data.csv`: example factor observations from all four time splits, with evaluator labels.",
                  "- `f1-comparison.svg`: mean seed-level F1 with 95% bootstrap intervals.",
                  "", "## External data already in this repository", "",
                  "The separate [NAB robust-residual report](../nab-robust-residual/README.md) evaluates 47 real series with published labeled windows and calibration-only thresholding. Its labels are not blind expert adjudication, it is a univariate task, and its scores are not directly comparable to this synthetic correlation test.",
                  "", "## Limitations", "", *[f"- {item}" for item in report["limitations"]], ""])
    markdown_path.write_text("\n".join(lines))
    _write_svg(summaries, root / "f1-comparison.svg")
    return json_path, markdown_path


def _fmt(summary: dict[str, Any], *, interval: bool = False) -> str:
    if interval:
        return f"{summary['mean']:+.3f} ({summary['lower']:+.3f} to {summary['upper']:+.3f})"
    return f"{summary['mean']:.3f} ({summary['lower']:.3f}–{summary['upper']:.3f})"


def _write_svg(summaries: dict[str, Any], path: Path) -> None:
    width, left, row_height = 880, 250, 55
    height = 80 + row_height * len(METHODS)
    scale = 500
    items = []
    for index, method in enumerate(METHODS):
        summary = summaries[method]["f1"]
        y = 52 + index * row_height
        x1 = left + max(0.0, summary["lower"]) * scale
        x2 = left + max(0.0, summary["upper"]) * scale
        x = left + summary["mean"] * scale
        items.append(f'<text x="14" y="{y + 5}" font-size="13" fill="#263a43">{method}</text>'
                     f'<line x1="{x1:.1f}" y1="{y}" x2="{x2:.1f}" y2="{y}" stroke="#497f91" stroke-width="3"/>'
                     f'<circle cx="{x:.1f}" cy="{y}" r="6" fill="#c06b48"/>'
                     f'<text x="{min(width - 55, x + 11):.1f}" y="{y + 5}" font-size="12" fill="#263a43">{summary["mean"]:.3f}</text>')
    ticks = ''.join(f'<text x="{left + value * scale - 10}" y="{height - 10}" font-size="11" fill="#60727a">{value:.1f}</text>'
                    for value in np.linspace(0, 1, 6))
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
           f'<rect width="100%" height="100%" fill="#f7f5ef"/><text x="14" y="24" font-size="16" fill="#263a43">Held-out F1 by lens (mean ± 95% seed bootstrap interval)</text>'
           + ''.join(items) + f'<line x1="{left}" y1="38" x2="{left}" y2="{height - 25}" stroke="#93a2a3"/>'
           + ticks + '</svg>')
    path.write_text(svg + "\n")


__all__ = ["run_correlation_benchmark", "write_correlation_benchmark"]
