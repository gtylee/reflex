# REFLEX training-only lens selection

- Synthetic scenarios: 6.
- Seeds: 3001–3030 (30 per scenario).
- Human review budget: 20 events per case, matched for both methods.
- Split: first 20 days fit, days 20 through 29 calibrate, day 30 onward held out.
- Lens choice uses training event structure and attributes. The threshold uses calibration scores. Evaluation labels are accessed only after predictions are frozen.

## Equal review budget results

| Method | Reviewed | Truth events | True positives | Precision | Recall | F1 | Mean case F1 (95% interval) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Selected lens | 3,600 | 23,878 | 2,672 | 0.742 | 0.112 | 0.194 | 0.230 (0.212–0.248) |
| Numeric baseline | 3,600 | 23,878 | 1,960 | 0.544 | 0.082 | 0.143 | 0.125 (0.109–0.141) |

## Frozen threshold results

These counts apply each method's calibration threshold without the review cap, so they show its natural alert volume.

| Method | Alerts | Truth events | True positives | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| Selected lens | 54,263 | 23,878 | 18,740 | 0.345 | 0.785 | 0.480 |
| Numeric baseline | 15,964 | 23,878 | 14,410 | 0.903 | 0.603 | 0.723 |

## Per-scenario F1

| Scenario | Selected lens F1 | Numeric baseline F1 | Lens choices |
|---|---:|---:|---|
| collateral_deterioration | 0.250 | 0.250 | multivariate_relationship × 30 |
| correlation_shift | 0.029 | 0.029 | multivariate_relationship × 30 |
| exposure_drift | 0.250 | 0.250 | multivariate_relationship × 1, robust_numeric_drift × 29 |
| rare_sequence | 0.431 | 0.002 | sequence_novelty × 30 |
| spurious_confounder | 0.185 | 0.185 | multivariate_relationship × 30 |
| unseen_mechanism | 0.235 | 0.031 | categorical_novelty × 30 |

## Limits

- The data is synthetic and the generated mechanisms are known to the benchmark author.
- The selector uses transparent structural rules, not a trained meta-model.
- The 99th-percentile threshold is recorded for operational use. The equal-budget comparison ranks the same 20 held-out events per method.
- The simple baseline is the maximum robust numeric deviation across training fields.

## Files

- [`selection-decisions.csv`](selection-decisions.csv): lens, rationale, training profile, and frozen threshold for each case.
- [`case-scores.csv`](case-scores.csv): metrics for each method and seed.
- [`threshold-scores.csv`](threshold-scores.csv): performance at each frozen threshold, without the review cap.
- [`reviewed-events.csv`](reviewed-events.csv): the exact 20 events each method sent to review.
- [`benchmark-report.json`](benchmark-report.json): complete protocol, per-scenario results, and limitations.
