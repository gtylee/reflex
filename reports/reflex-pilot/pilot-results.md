# REFLEX synthetic pilot

Repeated seeds: 3 (11, 23, 37).
Configuration: 40 entities, 30 days, 12 training epochs per seed.

| Lens | False discovery rate, mean (95% interval) | Hidden event recall, mean (95% interval) | Mechanisms recovered, mean (95% interval) |
|---|---:|---:|---:|
| stat.autoencoder | 0.037 (0.000–0.111) | 0.611 (0.333–0.786) | 1.33 (1.00–2.00) |
| stat.isolation_forest | 0.098 (0.000–0.182) | 0.694 (0.583–0.786) | 1.33 (1.00–2.00) |

Mean wall time: 0.83s (95% bootstrap interval 0.61–1.25s).

Each prediction covers one entity in a five-day window. It is counted as a true positive when the window contains a hidden event. Recall counts hidden events inside flagged windows.
Timing covers feature construction, model fitting, scoring, bounded investigations and evaluation. It excludes synthetic generation and event ingestion.

This pilot measures REFLEX statistical lenses against hidden synthetic labels. It does not compare variants A–E or establish that persistent state improves discovery.

All records are generated from actual runs. Ground truth is read only by the evaluator after scoring.
