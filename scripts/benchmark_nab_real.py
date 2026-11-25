"""Score REFLEX's statistical lenses on the real-world NAB series.

The first 10% of each series fits a model; the following 5% calibrates a
label-blind score percentile. Only later points are evaluated. The official
NAB window-scoring implementation is used with a fixed 98th-percentile alert
cutoff; NAB labels never tune model parameters or thresholds.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from reflex.core.discovery import Autoencoder


FEATURE_NAMES = ("value_z", "change_z", "trailing_mean_12", "trailing_std_12")
ALERT_THRESHOLD = 0.98
TRAIN_FRACTION = 0.10
CALIBRATION_FRACTION = 0.05
WINDOW_SIZE = 12
EPOCHS = 35


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.strip())


def read_series(path: Path) -> tuple[list[datetime], np.ndarray]:
    timestamps: list[datetime] = []
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not {"timestamp", "value"}.issubset(reader.fieldnames or ()):
            raise ValueError(f"Expected timestamp,value columns in {path}")
        for row in reader:
            timestamps.append(parse_time(row["timestamp"]))
            values.append(float(row["value"]))
    return timestamps, np.asarray(values, dtype=np.float32)


def make_features(values: np.ndarray, train_end: int) -> tuple[np.ndarray, dict[str, float]]:
    mean = float(np.mean(values[:train_end]))
    scale = float(np.std(values[:train_end]))
    if scale < 1e-8:
        scale = 1.0
    z = (values - mean) / scale
    delta = np.zeros_like(z)
    delta[1:] = z[1:] - z[:-1]
    trailing_mean = np.empty_like(z)
    trailing_std = np.empty_like(z)
    for index in range(len(z)):
        left = max(0, index - WINDOW_SIZE + 1)
        recent = z[left:index + 1]
        trailing_mean[index] = np.mean(recent)
        trailing_std[index] = np.std(recent)
    return np.column_stack((z, delta, trailing_mean, trailing_std)).astype(np.float32), {
        "raw_train_mean": mean,
        "raw_train_scale": scale,
    }


def empirical_percentile(calibration: np.ndarray, scores: np.ndarray) -> np.ndarray:
    ordered = np.sort(calibration)
    return np.searchsorted(ordered, scores, side="right") / (len(ordered) + 1)


def fit_and_score(
    name: str,
    features: np.ndarray,
    train_end: int,
    calibration_end: int,
    seed: int,
    series_index: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    scaler = StandardScaler().fit(features[:train_end])
    transformed = scaler.transform(features).astype(np.float32)
    fit_x = transformed[:train_end]
    calibration_x = transformed[train_end:calibration_end]
    if len(fit_x) < 20 or len(calibration_x) < 10:
        raise ValueError("Series is too short for the declared fit/calibration split")

    if name == "autoencoder":
        torch.manual_seed(seed * 1009 + series_index)
        model = Autoencoder(input_dim=fit_x.shape[1], latent_dim=2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
        fit_tensor = torch.from_numpy(fit_x)
        model.train()
        for _ in range(EPOCHS):
            optimizer.zero_grad()
            loss = torch.mean((model(fit_tensor) - fit_tensor) ** 2)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            all_tensor = torch.from_numpy(transformed)
            raw_scores = ((model(all_tensor) - all_tensor) ** 2).mean(dim=1).numpy()
        kind = "mean squared reconstruction error"
    elif name == "isolation_forest":
        model = IsolationForest(n_estimators=160, contamination=0.02,
                                random_state=seed * 1009 + series_index, n_jobs=1)
        model.fit(fit_x)
        raw_scores = -model.decision_function(transformed)
        kind = "negative Isolation Forest decision function"
    else:
        raise ValueError(name)

    calibration_scores = raw_scores[train_end:calibration_end]
    percentile_scores = np.zeros(len(raw_scores), dtype=np.float32)
    percentile_scores[calibration_end:] = empirical_percentile(
        calibration_scores, raw_scores[calibration_end:]).astype(np.float32)
    metadata = {
        "score_definition": kind,
        "calibration_cutoff_raw_score": float(np.quantile(calibration_scores, ALERT_THRESHOLD)),
        "calibration_points": int(len(calibration_scores)),
        **scaler_metadata(scaler),
    }
    return percentile_scores, metadata


def scaler_metadata(scaler: StandardScaler) -> dict[str, Any]:
    return {"feature_names": list(FEATURE_NAMES),
            "feature_means": [float(x) for x in scaler.mean_],
            "feature_scales": [float(x) for x in scaler.scale_]}


def read_windows(path: Path) -> dict[str, list[tuple[datetime, datetime]]]:
    raw = json.loads(path.read_text())
    return {name: [(parse_time(start), parse_time(end)) for start, end in windows]
            for name, windows in raw.items()}


def score_one(
    Sweeper: type,
    timestamps: list[datetime],
    scores: np.ndarray,
    windows: list[tuple[datetime, datetime]],
    profile: dict[str, Any],
) -> dict[str, Any]:
    sweep = Sweeper(probationPercent=0.0, costMatrix=profile["CostMatrix"])
    _, selected = sweep.scoreDataSet(timestamps, scores.tolist(), windows,
                                     "real-series", ALERT_THRESHOLD)
    _, null = sweep.scoreDataSet(timestamps, [0.0] * len(scores), windows,
                                 "null-baseline", ALERT_THRESHOLD)

    perfect_scores = [0.0] * len(scores)
    time_index = {timestamp: index for index, timestamp in enumerate(timestamps)}
    for start, _end in windows:
        perfect_scores[time_index[start]] = 1.0
    _, perfect = sweep.scoreDataSet(timestamps, perfect_scores, windows,
                                    "perfect-baseline", ALERT_THRESHOLD)
    if selected is None or null is None or perfect is None:
        raise RuntimeError("NAB scorer did not return a threshold score")

    predicted = [timestamps[i] for i, value in enumerate(scores) if value >= ALERT_THRESHOLD]
    detected_windows = sum(any(start <= point <= end for point in predicted)
                           for start, end in windows)
    false_alert_points = sum(not any(start <= point <= end for start, end in windows)
                             for point in predicted)
    return {
        "raw_score": float(selected.score),
        "null_score": float(null.score),
        "perfect_score": float(perfect.score),
        "normalized_score": (100.0 * (selected.score - null.score) /
                             (perfect.score - null.score)
                             if perfect.score != null.score else None),
        "detected_windows": int(detected_windows),
        "label_windows": int(len(windows)),
        "false_alert_points": int(false_alert_points),
        "alert_points": int(len(predicted)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nab-root", type=Path, default=Path(".reflex-nab/source"))
    parser.add_argument("--out", type=Path,
                        default=Path("reports/reflex-experiments/nab-real-lenses-review"))
    parser.add_argument("--seeds", default="11,23,37")
    args = parser.parse_args()

    nab_root = args.nab_root.resolve()
    sys.path.insert(0, str(nab_root))
    from nab.sweeper import Sweeper

    data_root = nab_root / "data"
    labels = read_windows(nab_root / "labels" / "combined_windows.json")
    profiles = json.loads((nab_root / "config" / "profiles.json").read_text())
    seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    real_files = sorted(path for category in data_root.iterdir()
                        if category.is_dir() and category.name.startswith("real")
                        for path in category.glob("*.csv"))
    if not real_files:
        raise FileNotFoundError(f"No real*.csv series found under {data_root}")

    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    profile_records: list[dict[str, Any]] = []
    series_records: list[dict[str, Any]] = []
    alert_records: list[dict[str, Any]] = []
    sample_records: list[dict[str, Any]] = []
    file_summary: dict[str, Any] = {}
    n_windows_scored = 0
    points_scored = 0
    category_counts: dict[str, int] = {}
    first_sample_window: tuple[datetime, datetime] | None = None

    for series_index, path in enumerate(real_files):
        relative = path.relative_to(data_root).as_posix()
        category_counts[path.parent.name] = category_counts.get(path.parent.name, 0) + 1
        timestamps, values = read_series(path)
        n = len(values)
        train_end = max(20, int(n * TRAIN_FRACTION))
        calibration_end = max(train_end + 10, int(n * (TRAIN_FRACTION + CALIBRATION_FRACTION)))
        calibration_end = min(calibration_end, n - 1)
        features, raw_stats = make_features(values, train_end)
        all_windows = labels.get(relative, [])
        cutoff_time = timestamps[calibration_end]
        test_windows = [(start, end) for start, end in all_windows if start >= cutoff_time]
        dropped_windows = len(all_windows) - len(test_windows)
        test_times = timestamps[calibration_end:]
        points_scored += len(test_times)
        n_windows_scored += len(test_windows)
        file_summary[relative] = {
            "rows": n,
            "train_points": train_end,
            "calibration_points": calibration_end - train_end,
            "test_points": len(test_times),
            "all_labeled_windows": len(all_windows),
            "test_labeled_windows": len(test_windows),
            "windows_before_or_overlapping_calibration": dropped_windows,
            "first_timestamp": timestamps[0].isoformat(sep=" "),
            "last_timestamp": timestamps[-1].isoformat(sep=" "),
        }

        for seed in seeds:
            for model_name in ("autoencoder", "isolation_forest"):
                scores, model_metadata = fit_and_score(
                    model_name, features, train_end, calibration_end, seed, series_index)
                heldout_scores = scores[calibration_end:]
                for profile_name, profile in profiles.items():
                    result = score_one(Sweeper, test_times, heldout_scores,
                                       test_windows, profile)
                    profile_records.append({
                        "seed": seed,
                        "lens": model_name,
                        "profile": profile_name,
                        "series": relative,
                        **result,
                    })
                    series_records.append({
                        "seed": seed,
                        "lens": model_name,
                    "profile": profile_name,
                        "series": relative,
                        "threshold_percentile": 98,
                        "threshold_score": ALERT_THRESHOLD,
                        **result,
                    })

                for local_index, score in enumerate(heldout_scores):
                    if score < ALERT_THRESHOLD:
                        continue
                    timestamp = test_times[local_index]
                    in_window = any(start <= timestamp <= end for start, end in test_windows)
                    alert_records.append({
                        "seed": seed,
                        "lens": model_name,
                        "series": relative,
                        "timestamp": timestamp.isoformat(sep=" "),
                        "value": float(values[calibration_end + local_index]),
                        "anomaly_percentile": float(score),
                        "inside_labeled_window": in_window,
                    })

        if relative == "realKnownCause/machine_temperature_system_failure.csv":
            first_sample_window = test_windows[0] if test_windows else None
            candidate_indices = set(range(min(12, n)))
            if first_sample_window:
                start_index = timestamps.index(first_sample_window[0])
                candidate_indices.update(range(max(0, start_index - 4),
                                               min(n, start_index + 6)))
            for i in sorted(candidate_indices):
                in_window = any(start <= timestamps[i] <= end for start, end in all_windows)
                sample_records.append({
                    "series": relative,
                    "timestamp": timestamps[i].isoformat(sep=" "),
                    "value": float(values[i]),
                    "in_labeled_anomaly_window": in_window,
                    "sample_section": "start_of_series" if i < 12 else "first_test_anomaly_window",
                })

    aggregate: dict[str, Any] = {}
    for lens in ("autoencoder", "isolation_forest"):
        aggregate[lens] = {}
        for profile_name in profiles:
            runs = []
            for seed in seeds:
                matched = [r for r in profile_records
                           if r["lens"] == lens and r["profile"] == profile_name and r["seed"] == seed]
                raw_score = sum(r["raw_score"] for r in matched)
                null_score = sum(r["null_score"] for r in matched)
                perfect_score = sum(r["perfect_score"] for r in matched)
                normalized = (100.0 * (raw_score - null_score) /
                              (perfect_score - null_score)
                              if perfect_score != null_score else None)
                runs.append({"seed": seed, "raw_score": raw_score,
                             "null_score": null_score, "perfect_score": perfect_score,
                             "normalized_score": normalized})
            vals = [r["normalized_score"] for r in runs if r["normalized_score"] is not None]
            aggregate[lens][profile_name] = {
                "mean_normalized_score": float(np.mean(vals)) if vals else None,
                "std_normalized_score": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                "per_seed": runs,
            }

    scores_csv = args.out / "per-series-scores.csv"
    score_columns = list(series_records[0].keys())
    with scores_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=score_columns)
        writer.writeheader(); writer.writerows(series_records)
    alerts_csv = args.out / "alerts.csv"
    alert_columns = list(alert_records[0].keys()) if alert_records else [
        "seed", "lens", "series", "timestamp", "value", "anomaly_percentile", "inside_labeled_window"]
    with alerts_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=alert_columns)
        writer.writeheader(); writer.writerows(alert_records)
    sample_csv = args.out / "sample-data.csv"
    with sample_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(sample_records[0].keys()))
        writer.writeheader(); writer.writerows(sample_records)

    commit = subprocess.check_output(["git", "-C", str(nab_root), "rev-parse", "HEAD"], text=True).strip()
    report = {
        "benchmark": "Numenta Anomaly Benchmark, real-world series only",
        "nab_repository_commit": commit,
        "dataset_files": len(real_files),
        "series_by_category": dict(sorted(category_counts.items())),
        "seeds": seeds,
        "rows_scored": points_scored,
        "labeled_windows_scored": n_windows_scored,
        "protocol": {
            "fit_fraction": TRAIN_FRACTION,
            "calibration_fraction": CALIBRATION_FRACTION,
            "test_fraction": 1.0 - TRAIN_FRACTION - CALIBRATION_FRACTION,
            "split": "chronological, independently within each series",
            "test_threshold": ALERT_THRESHOLD,
            "threshold_source": "98th percentile of each series/model calibration scores; no labels used for fitting or threshold selection",
            "features": list(FEATURE_NAMES),
            "trailing_window_points": WINDOW_SIZE,
            "autoencoder_epochs": EPOCHS,
            "isolation_forest_estimators": 160,
            "nab_score": "official NAB window-scoring implementation; normalized against no-alert and ideal baselines on this real-only held-out subset",
        },
        "aggregate_scores": aggregate,
        "series": file_summary,
        "alerts": {
            "alert_points": len(alert_records),
            "inside_labeled_windows": sum(bool(r["inside_labeled_window"]) for r in alert_records),
        },
        "limits": [
            "This is a real-world univariate time-series benchmark, not an FX transaction benchmark.",
            "NAB labels are used only by the scorer after scores are frozen.",
            "Scores use a fixed calibration-only 98th-percentile threshold; they are not directly comparable with published NAB leaderboard results that optimize thresholds against labels.",
            "Rows from NAB's artificial series are excluded.",
        ],
        "files": {
            "per_series_scores": scores_csv.name,
            "alerts": alerts_csv.name,
            "sample_data": sample_csv.name,
        },
    }
    report_path = args.out / "benchmark-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# NAB real-world series: REFLEX lens benchmark", "",
        f"- Series: {len(real_files)} real-world time series; artificial series excluded.",
        f"- Chronological splits per series: 10% fit / 5% calibration / 85% held-out.",
        f"- Held-out rows scored: {points_scored:,}; labeled windows after calibration: {n_windows_scored:,}.",
        f"- Alert threshold: top 2% of calibration scores per lens and series.",
        f"- NAB repository commit: `{commit}`.", "",
        "## Normalized NAB scores", "",
        "Scores are normalized to the no-alert and ideal baselines over this real-only held-out subset. Published leaderboard results may use label-optimized thresholds, so they are not directly comparable.",
        "", "| Lens | Standard | Reward low false positives | Reward low false negatives |",
        "|---|---:|---:|---:|",
    ]
    for lens in ("autoencoder", "isolation_forest"):
        vals = [aggregate[lens][profile]["mean_normalized_score"] for profile in profiles]
        devs = [aggregate[lens][profile]["std_normalized_score"] for profile in profiles]
        lines.append(f"| {lens.replace('_', ' ').title()} | {vals[0]:.1f} ± {devs[0]:.1f} | {vals[1]:.1f} ± {devs[1]:.1f} | {vals[2]:.1f} ± {devs[2]:.1f} |")
    lines += [
        "", "## What the data looks like", "",
        "Each row is one timestamp and one numeric measurement from a real series. The separate NAB label file marks the known anomaly windows; that label is not passed to either lens.",
        "", "| Dataset | Rows | Labeled windows |", "|---|---:|---:|",
    ]
    sample_name = "realKnownCause/machine_temperature_system_failure.csv"
    sample_meta = file_summary[sample_name]
    lines.append(f"| `{sample_name}` | {sample_meta['rows']:,} | {sample_meta['test_labeled_windows']} in held-out period |")
    lines += [
        "", "## Artifacts", "",
        "- [`sample-data.csv`](sample-data.csv): example rows, including the first held-out labeled window.",
        "- [`per-series-scores.csv`](per-series-scores.csv): scores by lens, series, profile, and seed.",
        "- [`alerts.csv`](alerts.csv): every point-level alert and whether it fell inside a labeled window.",
        "- [`benchmark-report.json`](benchmark-report.json): protocol, series counts, per-seed results, and limitations.",
        "", "This benchmark measures labeled time-series anomaly detection. It does not establish performance on FX data or multi-event workflows.",
    ]
    (args.out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({
        "report": str(report_path.resolve()),
        "readme": str((args.out / "README.md").resolve()),
        "sample_data": str(sample_csv.resolve()),
        "series": len(real_files),
        "rows_scored": points_scored,
        "labeled_windows_scored": n_windows_scored,
        "aggregate_scores": {lens: {profile: aggregate[lens][profile]["mean_normalized_score"]
                                     for profile in profiles}
                             for lens in aggregate},
        "alerts_csv": str(alerts_csv.resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
