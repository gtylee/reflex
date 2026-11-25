# REFLEX held-out synthetic benchmark

Seeds: 101–130 (30 per scenario).
Cases: 40 entities × 60 days. Variant: C.
Labels are held-out synthetic generator ground truth, not expert-labeled real cases.
Predictions are scored by exact event ID after calibration. Intervals resample seeds.

| Scenario | Seeds complete | Truth events | Precision | Recall | F1 | Evidence integrity |
|---|---:|---:|---:|---:|---:|---:|
| exposure_drift | 30/30 | 3000 | 0.998 (0.996–0.999) | 1.000 (1.000–1.000) | 0.999 (0.998–1.000) | 1.000 (1.000–1.000) |
| collateral_deterioration | 30/30 | 3000 | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) |
| rare_sequence | 30/30 | 300 | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) |
| correlation_shift | 30/30 | 4800 | 0.202 (0.195–0.207) | 0.427 (0.409–0.445) | 0.274 (0.265–0.282) | 1.000 (1.000–1.000) |
| spurious_confounder | 30/30 | 3935 | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) |
| unseen_mechanism | 30/30 | 3000 | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) | 1.000 (1.000–1.000) |

## Interpretation

- Seeds 101 onward are unseen generator seeds, not independently collected real-world cases.
- Labels are generated from known synthetic mechanisms and are not expert judgments.
- Only deterministic variant C is available in this run; it does not compare REFLEX with an agent baseline.
- Evidence integrity is an entity/time consistency check and does not establish that evidence is causally sufficient.
