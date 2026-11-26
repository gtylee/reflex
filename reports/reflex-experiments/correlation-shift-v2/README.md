# Paired correlation-shift benchmark

30 paired seeds (101–130), 40 entities, 60 days.
The baseline regime is used for training and calibration. The change begins halfway through the held-out period, leaving a stable held-out segment to measure false positives.
Thresholds and fitted values are determined without using evaluation labels.
Intervals are 95% percentile bootstrap intervals across seed-level results; F1 changes are paired by seed against the legacy lens.

| Lens | Precision | Recall | F1 | Stable held-out FPR | F1 change vs legacy |
|---|---:|---:|---:|---:|---:|
| legacy_product_quantile | 0.985 (0.982–0.989) | 0.419 (0.406–0.432) | 0.587 (0.575–0.600) | 0.006 (0.005–0.008) | +0.000 (+0.000 to +0.000) |
| calibrated_product_quantile | 0.976 (0.972–0.980) | 0.468 (0.457–0.480) | 0.632 (0.622–0.642) | 0.012 (0.010–0.014) | +0.045 (+0.031 to +0.059) |
| calibrated_rolling_correlation | 0.983 (0.976–0.989) | 0.992 (0.990–0.994) | 0.987 (0.984–0.990) | 0.018 (0.011–0.025) | +0.400 (+0.388 to +0.412) |
| calibrated_mahalanobis | 0.968 (0.963–0.974) | 0.343 (0.334–0.352) | 0.506 (0.496–0.516) | 0.011 (0.009–0.014) | -0.081 (-0.093 to -0.070) |
| calibrated_isolation_forest | 0.773 (0.735–0.811) | 0.045 (0.038–0.052) | 0.084 (0.072–0.097) | 0.013 (0.011–0.016) | -0.503 (-0.520 to -0.486) |

## Lens and threshold choices

- `legacy_product_quantile`: the original lower-tail `product_xy` quantile lens; cutoff fitted on training data.
- `calibrated_product_quantile`: same univariate feature, with a lower-tail cutoff from calibration data.
- `calibrated_rolling_correlation`: rolling 24-event correlation distance from the training correlation; threshold is the calibration 99th percentile.
- `calibrated_mahalanobis`: joint factor distance using training mean/covariance; threshold is the calibration 99th percentile.
- `calibrated_isolation_forest`: joint factor anomaly score fitted on training data; threshold is the calibration 1st score percentile.

## Reading the result

Each method sees identical event IDs for a seed, and each row in `paired-seed-results.csv` includes the shared input hash. F1 changes are paired by seed against the legacy lens.
The rolling-correlation lens cites a cross-entity window. This report evaluates detection only; the current same-entity evidence-integrity check does not assess cohort evidence correctly.
The generated labels mark the changed regime, so this measures detection of a known synthetic distribution change. It does not show that an alert is a real-world anomaly.

## Files

- `correlation-shift-report.json`: full protocol, diagnostics, hashes, and per-seed results.
- `paired-seed-results.csv`: per-seed paired candidate metrics and thresholds.
- `sample-data.csv`: example factor observations from all four time splits, with evaluator labels.
- `f1-comparison.svg`: mean seed-level F1 with 95% bootstrap intervals.

## External data already in this repository

The separate [NAB robust-residual report](../nab-robust-residual/README.md) evaluates 47 real series with published labeled windows and calibration-only thresholding. Its labels are not blind expert adjudication, it is a univariate task, and its scores are not directly comparable to this synthetic correlation test.

## Limitations

- This is a controlled synthetic test, not evidence on FX transactions or real market regimes.
- The benchmark evaluates known candidate lenses; it does not prove that an automated lens selector will choose correctly on unseen domains.
- The shifted regime occupies only half of each held-out period; the unchanged half provides a held-out false-positive check.
- The external NAB results use independently published anomaly windows, but no blind expert adjudication was performed and the data are univariate time series.
- Correlation evidence is cohort-wide. The current entity-local evidence-integrity check is not a suitable quality measure for these references and is not reported here.
- A/B/D/E agent runs remain unavailable; this experiment does not establish REFLEX superiority over a general-purpose agent.
