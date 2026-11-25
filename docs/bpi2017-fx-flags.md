# Reading REFLEX flags on BPI and applying the idea to FX

## What the BPI run contains

BPI Challenge 2017 is a loan-application workflow log. One application is a case; each timestamped row records an activity in that case, such as creating an application, completing a workflow step, or creating an offer. The source dataset contains 1,202,267 events across 31,509 applications, from January 2016 to February 2017. [Dataset page](https://data.4tu.nl/articles/dataset/BPI_Challenge_2017/12696884)

For example, an offer event for `Application_652823628` includes an offered amount of 20,000, monthly cost of 498.29, a credit score of 979, and 44 terms. The full 40-row sample trace is in [`sample-case.csv`](../reports/reflex-experiments/bpi2017-lens-review/sample-case.csv).

## What a statistical flag means

The Autoencoder and Isolation Forest produce scores from application-case features. In this run, features described the completed case: event count and duration, activity mix and repeats, resource variety, offers, and offer amounts. Outcome-status events and the `Accepted` and `Selected` fields were excluded from the model inputs.

The data was split by case completion time: 17,767 earlier cases for fitting, 7,723 for calibration, and 6,019 later cases for scoring. Each model's alert threshold was set at the 98th percentile of its calibration scores.

That percentile is an explicit **score cutoff**, but it is not an explicit **business rule**. There was no rule such as “flag every case with more than 20 missing-file calls” or “flag cases open longer than 90 days.” Instead, each model flags feature combinations that score as unusual compared with its training history. It does not know whether the behavior is harmful, erroneous, fraudulent, or simply rare.

The test period produced:

| Lens | Test cases flagged | Share of test cases |
|---|---:|---:|
| Autoencoder | 120 | 2.0% |
| Isolation Forest | 178 | 3.0% |
| Flagged by both | 62 | — |

The different counts show that the lenses do not agree on every unusual case. The Isolation Forest's later-period flag rate was about 3%, despite its cutoff being calibrated at 98th percentile; this run establishes the difference but does not explain its cause. The lenses' test score ranks had a correlation of 0.743. These numbers measure agreement and ranking behavior, not accuracy.

## What the reviewed cases suggest

Some flags point to potentially meaningful process friction:

- `Application_706230820`, flagged by both lenses, lasted about 142 days. Its trace included seven `W_Shortened completion` events and a gap of about 112 days between workflow events.
- `Application_114551747`, flagged by Isolation Forest, lasted about 98 days and contained 50 `W_Call incomplete files` events. The longest gap between those calls was about 26 days.

Other flags may be legitimate rare cases:

- `Application_968189026`, flagged by both, completed in about four minutes and ended in refusal. That may be a normal quick rejection rather than an operational problem.
- `Application_250165774` was cancelled but was not flagged. It had 35 feature events over about 16 days. Cancellation by itself therefore does not explain all the flags.

Across the flagged group, the median was 56 feature events, compared with 32 among unflagged cases. Flagged cases also had more distinct resources at the median: 11 versus 6. Cancellations and refusals were more common among flagged cases, but those outcome fields were not supplied as model features. These patterns make the flags useful review candidates; they do not establish that the cases are actual anomalies.

The dataset has no independent anomaly labels. We therefore cannot calculate precision, recall, or the share of flags that represent real process faults. The models also scored the full case after completion, so this is not an early-warning result.

## Where explicit rules fit

Use deterministic rules for controls whose meaning is known and must be enforced. Examples include a required approval before settlement, a notional or exposure ceiling, or a prohibited state transition. Such a rule should state the condition, threshold, and action clearly, and should be versioned and audited.

Use statistical flags to surface patterns that have not already been written as rules. Keep the score, threshold, feature values, supporting events, and model version with every observation. Route uncertain findings to review; do not treat the flag alone as a reason to block a transaction or take an external action.

## Applying this to FX transactions

REFLEX can represent FX data in the same event envelope: a trade or order ID is the subject, the event time records when a lifecycle change occurred, and attributes carry fields such as currency pair, side, quantity, rate, venue, and counterparty. Example event types might be `order.submitted`, `trade.filled`, `order.cancelled`, and `trade.settled`.

The BPI models and their scores should not be reused for FX. An FX run needs an FX-specific feature schema, a chronological fit/calibration/test split, and thresholds calibrated on historical FX activity. Known controls should be explicit rules; statistical lenses can then flag unusual price, size, timing, counterparty, or lifecycle patterns for review. Independently labeled FX incidents are needed to measure whether those flags identify real problems.

## Run artifacts

- [Run summary](../reports/reflex-experiments/bpi2017-lens-review/README.md)
- [Detailed lens report](../reports/reflex-experiments/bpi2017-lens-review/lens-report.json)
- [All flagged cases and scores](../reports/reflex-experiments/bpi2017-lens-review/flagged-cases.csv)
- [Sample application trace](../reports/reflex-experiments/bpi2017-lens-review/sample-case.csv)
