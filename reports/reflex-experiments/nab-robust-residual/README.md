# NAB real-world series: robust forecast-residual lens

- Series: 47 real-world time series; synthetic series excluded.
- Chronological split: 20% fit / 10% calibration / 70% held-out.
- Held-out rows scored: 224,866; held-out labeled windows: 78.
- Lens: causal robust forecast residual plus two-sided CUSUM; training-only log transform and seasonal phase are selected from each series' fit slice.
- Threshold: calibration median + 4.5 robust score scales, with a floor of 4.5; no NAB labels used to choose it.
- Held-out event alerts: 634; within labeled windows: 97; outside labeled windows: 537.

## Aggregate normalized NAB scores

Scores are normalized against no-alert and ideal baselines on this real-only held-out subset; they are not directly comparable to leaderboard scores.

| NAB profile | Normalized score |
|---|---:|
| standard | 25.9 |
| reward_low_FP_rate | -10.9 |
| reward_low_FN_rate | 39.1 |

## Review files

- [`training-diagnostics.csv`](training-diagnostics.csv): fit-only cadence, trend-cycle, and residual-scale diagnostics.
- [`thresholds.csv`](thresholds.csv): each series' unlabeled calibration threshold.
- [`per-series-scores.csv`](per-series-scores.csv): NAB results by series and profile.
- [`alerts.csv`](alerts.csv): frozen event alerts and their scores.
- [`sample-data.csv`](sample-data.csv): example measurements, split assignment, score, and final label for held-out rows.

Labels are used only after fitting, cycle selection, threshold calibration, and alert generation are complete.
Outside-window alerts are not necessarily confirmed false positives because NAB labels known windows, not every unusual event.
This univariate time-series run does not establish performance on FX transaction records or multi-event workflows.
