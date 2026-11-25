"""Run a label-blind robust forecasting-residual lens on real NAB series.

For each series, the first 20% is used to inspect cadence, select an optional
repeat cycle, and estimate residual behavior. The next 10% calibrates a robust
score threshold. Remaining points are scored chronologically. NAB labels are
loaded only after all model scores and alerts have been frozen.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


FIT_FRACTION = 0.20
CALIBRATION_FRACTION = 0.10
LOCAL_LOOKBACK_SECONDS = 60 * 60
MIN_LOCAL_LOOKBACK_POINTS = 6
MAX_LOCAL_LOOKBACK_POINTS = 48
MIN_CYCLE_SECONDS = 6 * 60 * 60
MIN_CYCLE_CORRELATION = 0.45
MIN_CYCLES_TO_SELECT = 3
THRESHOLD_ROBUST_SIGMAS = 4.5
THRESHOLD_FLOOR = 4.5
RELEASE_RATIO = 0.5
RELEASE_POINTS = 3
NAB_ALERT_THRESHOLD = 0.98


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
    if not values:
        raise ValueError(f"No observations in {path}")
    return timestamps, np.asarray(values, dtype=np.float64)


def robust_scale(values: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    center = float(np.median(values))
    deviations = np.abs(values - center)
    mad_scale = float(1.4826 * np.median(deviations))
    # The 80th absolute-deviation quantile helps when measurements are
    # quantized: a narrow pile of repeated values can make MAD much smaller
    # than the ordinary variation visible in the fit slice.
    q80_scale = float(np.quantile(deviations, 0.80) / 1.2816)
    if max(mad_scale, q80_scale) > 1e-12:
        return max(mad_scale, q80_scale)
    q25, q75 = np.quantile(values, [0.25, 0.75])
    iqr_scale = float((q75 - q25) / 1.349)
    if iqr_scale > 1e-12:
        return iqr_scale
    q90_scale = float(np.quantile(deviations, 0.90) / 1.645)
    if q90_scale > 1e-12:
        return q90_scale
    # Sparse series can have a large point mass at zero, making every central
    # quantile zero. Use the typical nonzero deviation as a scale estimate.
    nonzero = deviations[deviations > 1e-12]
    if len(nonzero):
        return float(np.median(nonzero) / 0.6745)
    return 0.0


def choose_transform(train_values: np.ndarray) -> str:
    """Use log1p only for nonnegative series with strong skew or many zeros."""
    if np.min(train_values) < 0:
        return "identity"
    zero_fraction = float(np.mean(train_values == 0))
    spread = float(np.std(train_values))
    if spread <= 1e-12:
        skew = 0.0
    else:
        centered = train_values - float(np.mean(train_values))
        skew = float(np.mean(centered ** 3) / spread ** 3)
    return "log1p" if zero_fraction >= 0.05 or skew >= 2.0 else "identity"


def cadence_seconds(timestamps: list[datetime]) -> float:
    gaps = np.asarray([
        (right - left).total_seconds()
        for left, right in zip(timestamps, timestamps[1:])
        if (right - left).total_seconds() > 0
    ], dtype=np.float64)
    return float(np.median(gaps)) if len(gaps) else 0.0


def local_lookback_points(cadence: float) -> int:
    if cadence <= 0:
        return 12
    return int(np.clip(round(LOCAL_LOOKBACK_SECONDS / cadence),
                       MIN_LOCAL_LOOKBACK_POINTS, MAX_LOCAL_LOOKBACK_POINTS))


def detrended_acf(values: np.ndarray, lag: int) -> float:
    if lag <= 0 or len(values) <= 3 * lag:
        return float("nan")
    x = np.asarray(values, dtype=np.float64)
    t = np.arange(len(x), dtype=np.float64)
    centered_t = t - np.mean(t)
    denom = float(np.dot(centered_t, centered_t))
    trend = float(np.dot(centered_t, x - np.mean(x)) / denom) if denom else 0.0
    residual = x - (np.mean(x) + trend * centered_t)
    left, right = residual[:-lag], residual[lag:]
    if np.std(left) < 1e-12 or np.std(right) < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def choose_cycle(values: np.ndarray, cadence: float) -> tuple[int | None, float]:
    """Choose a repeated period only when training contains 3+ cycles and a clear ACF peak."""
    n = len(values)
    max_lag = n // MIN_CYCLES_TO_SELECT
    if max_lag < 3:
        return None, float("nan")
    min_lag = max(6, int(round(MIN_CYCLE_SECONDS / cadence))) if cadence > 0 else 6
    min_lag = min(min_lag, max_lag)
    candidates: list[tuple[float, int]] = []
    acfs: dict[int, float] = {}
    for lag in range(min_lag, max_lag + 1):
        value = detrended_acf(values, lag)
        if np.isfinite(value):
            acfs[lag] = value
    for lag, value in acfs.items():
        left = acfs.get(lag - 1, -1.0)
        right = acfs.get(lag + 1, -1.0)
        if value >= left and value >= right and value >= MIN_CYCLE_CORRELATION:
            candidates.append((value, lag))
    if not candidates:
        return None, max(acfs.values(), default=float("nan"))
    # Prefer the clearest local cycle peak. The period remains in sample units;
    # cadence is stored separately so the choice can be interpreted in time.
    strength, lag = max(candidates)
    return lag, float(strength)


def forecast_residuals(values: np.ndarray, cycle: int | None,
                       lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Causal median forecast; use a seasonal phase only after 3 prior cycles."""
    n = len(values)
    baselines = np.full(n, np.nan, dtype=np.float64)
    residuals = np.full(n, np.nan, dtype=np.float64)
    for index in range(1, n):
        baseline: float
        if cycle and index >= 3 * cycle:
            prior_same_phase = values[index - cycle * np.arange(1, 4)]
            baseline = float(np.median(prior_same_phase))
        else:
            left = max(0, index - lookback)
            baseline = float(np.median(values[left:index]))
        baselines[index] = baseline
        residuals[index] = values[index] - baseline
    return baselines, residuals


def residual_scores(values: np.ndarray, residuals: np.ndarray,
                    fit_end: int, global_scale: float) -> np.ndarray:
    """Combine robust one-step residuals with a two-sided CUSUM for shifts."""
    scores = np.zeros(len(values), dtype=np.float64)
    positive_cusum = 0.0
    negative_cusum = 0.0
    floor = max(global_scale * 0.35, np.finfo(np.float64).eps *
                max(1.0, float(np.max(np.abs(values[:fit_end])))))
    for index in range(1, len(values)):
        residual = residuals[index]
        if not np.isfinite(residual):
            continue
        left = max(1, index - 12)
        recent = residuals[left:index]
        recent = recent[np.isfinite(recent)]
        local_scale = robust_scale(recent) if len(recent) >= 5 else 0.0
        scale = max(global_scale, local_scale, floor)
        z = float(residual / scale)
        positive_cusum = max(0.0, positive_cusum + z - 0.5)
        negative_cusum = max(0.0, negative_cusum - z - 0.5)
        scores[index] = max(abs(z), positive_cusum, negative_cusum)
    return scores


def choose_threshold(calibration_scores: np.ndarray) -> tuple[float, float, float]:
    finite = calibration_scores[np.isfinite(calibration_scores)]
    if not len(finite):
        return THRESHOLD_FLOOR, 0.0, 0.0
    center = float(np.median(finite))
    scale = robust_scale(finite)
    threshold = max(THRESHOLD_FLOOR, center + THRESHOLD_ROBUST_SIGMAS * scale)
    return float(threshold), center, scale


def make_events(scores: np.ndarray, threshold: float) -> tuple[np.ndarray, list[int]]:
    """Emit one event at threshold crossing; re-arm only after a quiet run."""
    events = np.zeros(len(scores), dtype=np.float64)
    starts: list[int] = []
    active = False
    quiet = 0
    for index, score in enumerate(scores):
        if not active and score >= threshold:
            events[index] = 1.0
            starts.append(index)
            active = True
            quiet = 0
        elif active:
            if score < threshold * RELEASE_RATIO:
                quiet += 1
                if quiet >= RELEASE_POINTS:
                    active = False
                    quiet = 0
            else:
                quiet = 0
    return events, starts


def read_windows(path: Path) -> dict[str, list[tuple[datetime, datetime]]]:
    raw = json.loads(path.read_text())
    return {name: [(parse_time(start), parse_time(end)) for start, end in windows]
            for name, windows in raw.items()}


def score_one(Sweeper: type, timestamps: list[datetime], alerts: np.ndarray,
              windows: list[tuple[datetime, datetime]], profile: dict[str, Any]) -> dict[str, Any]:
    sweep = Sweeper(probationPercent=0.0, costMatrix=profile["CostMatrix"])
    _, selected = sweep.scoreDataSet(timestamps, alerts.tolist(), windows,
                                     "robust-residual", NAB_ALERT_THRESHOLD)
    _, null = sweep.scoreDataSet(timestamps, [0.0] * len(alerts), windows,
                                 "null-baseline", NAB_ALERT_THRESHOLD)
    perfect_alerts = np.zeros(len(alerts), dtype=np.float64)
    index_by_time = {timestamp: index for index, timestamp in enumerate(timestamps)}
    for start, _end in windows:
        if start in index_by_time:
            perfect_alerts[index_by_time[start]] = 1.0
    _, perfect = sweep.scoreDataSet(timestamps, perfect_alerts.tolist(), windows,
                                    "perfect-baseline", NAB_ALERT_THRESHOLD)
    if selected is None or null is None or perfect is None:
        raise RuntimeError("NAB scorer did not return a threshold score")
    detected = sum(np.any(alerts[(np.asarray(timestamps) >= start) &
                                 (np.asarray(timestamps) <= end)] >= NAB_ALERT_THRESHOLD)
                   for start, end in windows)
    return {
        "raw_score": float(selected.score),
        "null_score": float(null.score),
        "perfect_score": float(perfect.score),
        "normalized_score": (100.0 * (selected.score - null.score) /
                             (perfect.score - null.score)
                             if perfect.score != null.score else None),
        "detected_windows": int(detected),
        "labeled_windows": int(len(windows)),
        "event_alerts": int(np.sum(alerts >= NAB_ALERT_THRESHOLD)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nab-root", type=Path, default=Path(".reflex-nab/source"))
    parser.add_argument("--out", type=Path,
                        default=Path("reports/reflex-experiments/nab-robust-residual"))
    args = parser.parse_args()
    nab_root = args.nab_root.resolve()
    sys.path.insert(0, str(nab_root))
    from nab.sweeper import Sweeper

    data_root = nab_root / "data"
    real_files = sorted(path for category in data_root.iterdir()
                        if category.is_dir() and category.name.startswith("real")
                        for path in category.glob("*.csv"))
    if not real_files:
        raise FileNotFoundError(f"No real*.csv series found under {data_root}")
    profiles = json.loads((nab_root / "config" / "profiles.json").read_text())

    args.out.mkdir(parents=True, exist_ok=True)
    diagnostics: list[dict[str, Any]] = []
    threshold_records: list[dict[str, Any]] = []
    alert_records: list[dict[str, Any]] = []
    frozen: dict[str, dict[str, Any]] = {}
    category_counts: dict[str, int] = {}

    # First pass uses only observations. The label file is deliberately not read
    # until scores, thresholds, and alert events are frozen for every series.
    for path in real_files:
        relative = path.relative_to(data_root).as_posix()
        category_counts[path.parent.name] = category_counts.get(path.parent.name, 0) + 1
        timestamps, values = read_series(path)
        n = len(values)
        fit_end = max(30, int(n * FIT_FRACTION))
        calibration_end = max(fit_end + 20, int(n * (FIT_FRACTION + CALIBRATION_FRACTION)))
        calibration_end = min(calibration_end, n - 1)
        fit_values = values[:fit_end]
        transform = choose_transform(fit_values)
        model_values = np.log1p(np.maximum(values, 0.0)) if transform == "log1p" else values.copy()
        model_fit_values = model_values[:fit_end]
        cadence = cadence_seconds(timestamps)
        lookback = local_lookback_points(cadence)
        cycle, cycle_strength = choose_cycle(model_fit_values, cadence)
        baselines, residuals = forecast_residuals(model_values, cycle, lookback)
        # Scale and score configuration are learned from the fit segment only.
        fit_residuals = residuals[1:fit_end]
        fit_residuals = fit_residuals[np.isfinite(fit_residuals)]
        global_scale = robust_scale(fit_residuals)
        if global_scale <= 0:
            differences = np.diff(model_fit_values)
            global_scale = robust_scale(differences)
        global_scale = max(global_scale, np.finfo(np.float64).eps)
        scores = residual_scores(model_values, residuals, fit_end, global_scale)
        threshold, calibration_median, calibration_mad = choose_threshold(
            scores[fit_end:calibration_end])
        all_events, event_indices = make_events(scores, threshold)
        test_alerts = np.zeros(n - calibration_end, dtype=np.float64)
        for index in event_indices:
            if index >= calibration_end:
                test_alerts[index - calibration_end] = 1.0
                alert_records.append({
                    "series": relative,
                    "category": path.parent.name,
                    "timestamp": timestamps[index].isoformat(sep=" "),
                    "value": float(values[index]),
                    "score": float(scores[index]),
                    "threshold": threshold,
                })
        frozen[relative] = {
            "timestamps": timestamps[calibration_end:],
            "alerts": test_alerts,
            "test_start": timestamps[calibration_end],
            "rows": n,
            "fit_points": fit_end,
            "calibration_points": calibration_end - fit_end,
            "test_points": n - calibration_end,
            "value_transform": transform,
            "cycle_points": cycle,
            "cycle_hours": cycle * cadence / 3600 if cycle else None,
            "cycle_strength": cycle_strength,
            "cadence_seconds": cadence,
            "lookback_points": lookback,
            "fit_residual_scale": global_scale,
            "calibration_score_median": calibration_median,
            "calibration_score_robust_scale": calibration_mad,
            "threshold": threshold,
            "all_event_count": len(event_indices),
            "test_event_count": int(np.sum(test_alerts)),
            "fit_end": fit_end,
            "calibration_end": calibration_end,
            "sample_timestamps": timestamps,
            "sample_values": values,
            "sample_scores": scores,
            "sample_events": all_events,
        }
        diagnostics.append({
            "series": relative,
            "category": path.parent.name,
            "rows": n,
            "fit_points": fit_end,
            "calibration_points": calibration_end - fit_end,
            "test_points": n - calibration_end,
            "value_transform": transform,
            "cadence_seconds": cadence,
            "local_lookback_points": lookback,
            "selected_cycle_points": cycle or "",
            "selected_cycle_hours": (cycle * cadence / 3600) if cycle else "",
            "cycle_acf_strength": cycle_strength,
            "fit_residual_robust_scale": global_scale,
        })
        threshold_records.append({
            "series": relative,
            "category": path.parent.name,
            "value_transform": transform,
            "selected_cycle_points": cycle or "",
            "selected_cycle_hours": (cycle * cadence / 3600) if cycle else "",
            "calibration_points": calibration_end - fit_end,
            "calibration_score_median": calibration_median,
            "calibration_score_robust_scale": calibration_mad,
            "robust_sigma_multiplier": THRESHOLD_ROBUST_SIGMAS,
            "threshold_floor": THRESHOLD_FLOOR,
            "score_threshold": threshold,
            "test_event_alerts": int(np.sum(test_alerts)),
        })

    # Labels are now read solely for evaluation of the frozen predictions.
    labels = read_windows(nab_root / "labels" / "combined_windows.json")
    score_records: list[dict[str, Any]] = []
    series_meta: dict[str, Any] = {}
    aggregate_raw: dict[str, dict[str, list[float]]] = {
        profile_name: {field: [] for field in ("raw", "null", "perfect")}
        for profile_name in profiles
    }
    total_windows = 0
    total_points = 0
    test_events = 0
    known_window_events = 0
    outside_window_events = 0
    for relative, result in frozen.items():
        timestamps = result["timestamps"]
        alerts = result["alerts"]
        windows = [(start, end) for start, end in labels.get(relative, [])
                   if start >= result["test_start"]]
        total_points += len(timestamps)
        total_windows += len(windows)
        test_events += int(np.sum(alerts))
        in_windows = []
        for index, alert in enumerate(alerts):
            if alert >= NAB_ALERT_THRESHOLD:
                point = timestamps[index]
                inside = any(start <= point <= end for start, end in windows)
                in_windows.append(inside)
                if inside:
                    known_window_events += 1
                else:
                    outside_window_events += 1
        series_meta[relative] = {
            key: value for key, value in result.items()
            if key not in {"timestamps", "alerts", "sample_timestamps", "sample_values",
                           "sample_scores", "sample_events", "fit_end", "calibration_end",
                           "test_start"}
        }
        for profile_name, profile in profiles.items():
            scored = score_one(Sweeper, timestamps, alerts, windows, profile)
            score_records.append({
                "series": relative,
                "category": relative.split("/", 1)[0],
                "profile": profile_name,
                **scored,
                "outside_labeled_window_events": int(sum(not value for value in in_windows)),
            })
            aggregate_raw[profile_name]["raw"].append(scored["raw_score"])
            aggregate_raw[profile_name]["null"].append(scored["null_score"])
            aggregate_raw[profile_name]["perfect"].append(scored["perfect_score"])

    aggregate: dict[str, Any] = {}
    for profile_name, values in aggregate_raw.items():
        raw = float(sum(values["raw"]))
        null = float(sum(values["null"]))
        perfect = float(sum(values["perfect"]))
        aggregate[profile_name] = {
            "raw_score": raw,
            "no_alert_score": null,
            "ideal_score": perfect,
            "normalized_score": (100.0 * (raw - null) / (perfect - null)
                                 if perfect != null else None),
        }

    # Short sample from the same machine-temperature series for transparent inspection.
    sample_name = "realKnownCause/machine_temperature_system_failure.csv"
    sample_rows: list[dict[str, Any]] = []
    sample = frozen.get(sample_name)
    if sample:
        ts = sample["sample_timestamps"]
        vals = sample["sample_values"]
        end = sample["calibration_end"]
        fit = sample["fit_end"]
        label_windows = labels.get(sample_name, [])
        candidates = set(range(min(12, len(vals))))
        next_window = next((start for start, _ in label_windows if start >= ts[end]), None)
        if next_window:
            start_index = ts.index(next_window)
            candidates.update(range(max(0, start_index - 4), min(len(vals), start_index + 6)))
        for index in sorted(candidates):
            is_test = index >= end
            in_window = any(start <= ts[index] <= stop for start, stop in label_windows) if is_test else ""
            sample_rows.append({
                "timestamp": ts[index].isoformat(sep=" "),
                "value": float(vals[index]),
                "split": "fit" if index < fit else "calibration" if index < end else "held-out",
                "in_labeled_anomaly_window": in_window,
                "robust_score": float(sample["sample_scores"][index]),
                "alert_event": bool(sample["sample_events"][index]),
            })

    out = args.out
    write_csv(out / "training-diagnostics.csv", diagnostics)
    write_csv(out / "thresholds.csv", threshold_records)
    write_csv(out / "per-series-scores.csv", score_records)
    write_csv(out / "alerts.csv", alert_records)
    write_csv(out / "sample-data.csv", sample_rows)
    commit = subprocess.check_output(["git", "-C", str(nab_root), "rev-parse", "HEAD"],
                                     text=True).strip()
    report = {
        "benchmark": "NAB real-world series, robust forecast-residual lens",
        "nab_repository_commit": commit,
        "series_count": len(real_files),
        "series_by_category": dict(sorted(category_counts.items())),
        "rows_scored": total_points,
        "labeled_windows_scored": total_windows,
        "threshold_selection": {
            "fit_fraction": FIT_FRACTION,
            "calibration_fraction": CALIBRATION_FRACTION,
            "test_fraction": 1.0 - FIT_FRACTION - CALIBRATION_FRACTION,
            "threshold_rule": "max(floor, calibration median + 4.5 robust standard deviations)",
            "threshold_floor": THRESHOLD_FLOOR,
            "calibration_robust_scale": "1.4826 x median absolute deviation of causal composite scores",
            "labels_used_for_selection": False,
        },
        "lens": {
            "value_transform": "training-only log1p for nonnegative, zero-heavy or strongly skewed series; identity otherwise",
            "forecast": "prior rolling median; use same-phase median of 3 prior cycles when a strong cycle is detected in fit data and at least 3 cycles are available",
            "cycle_selection": "detrended training autocorrelation local peak >= 0.45; period must be at least 6 hours and fit data must contain at least 3 cycles",
            "score": "maximum of absolute robust forecast residual and two-sided CUSUM (drift allowance 0.5)",
            "residual_scale": "fit residual scale is robust, with an 80th absolute-deviation fallback for quantized series",
            "event_policy": f"emit one alert at threshold crossing; re-arm after {RELEASE_POINTS} points below {RELEASE_RATIO:.1f} x threshold",
        },
        "evaluation": {
            "method": "official NAB window scorer; labels loaded only after frozen scores and alerts",
            "normalized_scores": aggregate,
            "event_alerts": test_events,
            "alerts_inside_labeled_windows": known_window_events,
            "alerts_outside_labeled_windows": outside_window_events,
            "outside_window_caveat": "Outside a labeled window means NAB did not label that point as anomalous; it is not proof the alert is false.",
        },
        "series": series_meta,
        "files": {
            "training_diagnostics": "training-diagnostics.csv",
            "thresholds": "thresholds.csv",
            "per_series_scores": "per-series-scores.csv",
            "alerts": "alerts.csv",
            "sample_data": "sample-data.csv",
        },
    }
    (out / "benchmark-report.json").write_text(json.dumps(report, indent=2) + "\n",
                                                encoding="utf-8")
    lines = [
        "# NAB real-world series: robust forecast-residual lens", "",
        f"- Series: {len(real_files)} real-world time series; synthetic series excluded.",
        f"- Chronological split: {FIT_FRACTION:.0%} fit / {CALIBRATION_FRACTION:.0%} calibration / {1 - FIT_FRACTION - CALIBRATION_FRACTION:.0%} held-out.",
        f"- Held-out rows scored: {total_points:,}; held-out labeled windows: {total_windows:,}.",
        "- Lens: causal robust forecast residual plus two-sided CUSUM; training-only log transform and seasonal phase are selected from each series' fit slice.",
        "- Threshold: calibration median + 4.5 robust score scales, with a floor of 4.5; no NAB labels used to choose it.",
        f"- Held-out event alerts: {test_events:,}; within labeled windows: {known_window_events:,}; outside labeled windows: {outside_window_events:,}.",
        "", "## Aggregate normalized NAB scores", "",
        "Scores are normalized against no-alert and ideal baselines on this real-only held-out subset; they are not directly comparable to leaderboard scores.",
        "", "| NAB profile | Normalized score |", "|---|---:|",
    ]
    for profile_name, values in aggregate.items():
        lines.append(f"| {profile_name} | {values['normalized_score']:.1f} |")
    lines += [
        "", "## Review files", "",
        "- [`training-diagnostics.csv`](training-diagnostics.csv): fit-only cadence, trend-cycle, and residual-scale diagnostics.",
        "- [`thresholds.csv`](thresholds.csv): each series' unlabeled calibration threshold.",
        "- [`per-series-scores.csv`](per-series-scores.csv): NAB results by series and profile.",
        "- [`alerts.csv`](alerts.csv): frozen event alerts and their scores.",
        "- [`sample-data.csv`](sample-data.csv): example measurements, split assignment, score, and final label for held-out rows.",
        "", "Labels are used only after fitting, cycle selection, threshold calibration, and alert generation are complete.",
        "Outside-window alerts are not necessarily confirmed false positives because NAB labels known windows, not every unusual event.",
        "This univariate time-series run does not establish performance on FX transaction records or multi-event workflows.",
    ]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str((out / "benchmark-report.json").resolve()),
        "readme": str((out / "README.md").resolve()),
        "series": len(real_files),
        "rows_scored": total_points,
        "labeled_windows": total_windows,
        "event_alerts": test_events,
        "inside_labeled_windows": known_window_events,
        "outside_labeled_windows": outside_window_events,
        "aggregate_scores": aggregate,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
