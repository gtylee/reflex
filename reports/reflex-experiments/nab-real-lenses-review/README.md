# NAB real-world series: REFLEX lens benchmark

- Series: 47 real-world time series; artificial series excluded.
- Chronological splits per series: 10% fit / 5% calibration / 85% held-out.
- Held-out rows scored: 273,050; labeled windows after calibration: 101.
- Alert threshold: top 2% of calibration scores per lens and series.
- NAB repository commit: `ea702d75cc2258d9d7dd35ca8e5e2539d71f3140`.

## Normalized NAB scores

Scores are normalized to the no-alert and ideal baselines over this real-only held-out subset. Published leaderboard results may use label-optimized thresholds, so they are not directly comparable.

| Lens | Standard | Reward low false positives | Reward low false negatives |
|---|---:|---:|---:|
| Autoencoder | -840.1 ± 36.8 | -1763.8 ± 74.2 | -531.1 ± 24.5 |
| Isolation Forest | -804.3 ± 20.3 | -1696.1 ± 40.4 | -506.1 ± 13.6 |

## What the data looks like

Each row is one timestamp and one numeric measurement from a real series. The separate NAB label file marks the known anomaly windows; that label is not passed to either lens.

| Dataset | Rows | Labeled windows |
|---|---:|---:|
| `realKnownCause/machine_temperature_system_failure.csv` | 22,695 | 3 in held-out period |

## Artifacts

- [`sample-data.csv`](sample-data.csv): example rows, including the first held-out labeled window.
- [`per-series-scores.csv`](per-series-scores.csv): scores by lens, series, profile, and seed.
- [`alerts.csv`](alerts.csv): every point-level alert and whether it fell inside a labeled window.
- [`benchmark-report.json`](benchmark-report.json): protocol, series counts, per-seed results, and limitations.

This benchmark measures labeled time-series anomaly detection. It does not establish performance on FX data or multi-event workflows.
