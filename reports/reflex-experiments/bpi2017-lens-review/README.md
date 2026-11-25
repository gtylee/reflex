# BPI Challenge 2017: REFLEX lens exploration

- Cases: 31,509; source events: 1,202,267
- Outcome-status events excluded from model features: 57,005
- Date span: 2016-01-01 to 2017-02-01
- Temporal case split: 17,767 train / 7,723 calibrate / 6,019 test
- Flagged at held-out calibration 98th percentile: Autoencoder 120; Isolation Forest 178; overlap 62
- Test score rank correlation: 0.743

This is an unsupervised case-level outlier review. The dataset does not provide anomaly truth labels, so precision/recall are not reported. Cases are scored after completion, not as early warnings.

## Highest-scoring cases

| Lens | Case | Percentile | Events | Duration (h) | Offers | Final activity |
|---|---|---:|---:|---:|---:|---|
| Autoencoder | Application_706230820 | 100.0 | 16 | 3419.68 | 1 | O_Cancelled |
| Autoencoder | Application_249437501 | 100.0 | 46 | 739.11 | 1 | O_Refused |
| Autoencoder | Application_1034800486 | 99.9 | 60 | 1126.17 | 1 | O_Refused |
| Autoencoder | Application_1220541878 | 99.9 | 87 | 1127.49 | 1 | O_Refused |
| Autoencoder | Application_512079496 | 99.9 | 20 | 767.39 | 1 | O_Cancelled |
| Isolation Forest | Application_968189026 | 100.0 | 7 | 0.06 | 1 | O_Refused |
| Isolation Forest | Application_765444982 | 100.0 | 11 | 38.23 | 1 | O_Cancelled |
| Isolation Forest | Application_2061502784 | 100.0 | 27 | 1319.9 | 4 | O_Cancelled |
| Isolation Forest | Application_739325744 | 99.9 | 110 | 3349.88 | 2 | O_Cancelled |
| Isolation Forest | Application_2037905525 | 99.9 | 7 | 0.12 | 1 | O_Cancelled |
