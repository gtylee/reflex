# BPI Challenge 2017 process-profile example

This example uses the shared REFLEX package builder to compare two
application-owned candidates: a pooled robust process profile and a clustered
profile. It demonstrates that a process lens can be added without editing
`src/reflex/core`.

## What the lens sees

The adapter groups events by application case and emits one feature row per
completed case. Its six features describe event count, elapsed process hours,
activity and resource diversity, offer count, and repeated activities. The
feature list and candidate code live in `lens.py`; XES parsing and case
aggregation live in `build.py`.

The five outcome-status activities (`A_Cancelled`, `A_Denied`, `O_Accepted`,
`O_Cancelled`, and `O_Refused`) are excluded from feature values and supporting
evidence. Their timestamps can identify when the case is complete, but the
outcome itself is not a model input. Feature evidence points to the remaining
process events.
The resulting score is post-completion review, not an early warning.

## Training and review

Cases are split chronologically by completion time, keeping ties together.
About 80% of the first training block fits candidate models; its later 20%
selects the candidate. The next block calibrates a 98th-percentile cutoff. The
remaining cases are scored only after the package is saved and reloaded.
Selection uses median normalized reconstruction error for both candidates.

The original log has outcomes but no independent anomaly labels. The example
reports flagged counts and evidence for inspection; it does not report
precision or recall and does not treat a case outcome as anomaly truth.

## Run result

The local run used 31,509 cases and 1,202,267 source events. It used
1,145,262 process events in feature evidence and excluded 57,005 final status
events. Keeping equal completion timestamps together produced 14,217 fit cases,
3,554 candidate-selection cases, 7,720 calibration cases, and 6,018 held-out
cases. The clustered candidate won with a median normalized
reconstruction-error proxy of 0.386, compared with 0.687 for the pooled
candidate. The 98th-percentile calibration cutoff was 0.940; it flagged 154
calibration cases and 165 held-out cases.

One highest-scoring held-out case was `Application_2011518740` (score 3.682,
cluster 1): 54 process events, 16 distinct activities, 8 resources, 38 repeated
activity instances, one offer, and 6,866 hours between its first and last
feature events. This is a process-profile lead for manual review. The score
does not say the application is fraudulent, erroneous, or otherwise bad.

## Run

The raw file is kept locally at `.reflex-bpi2017/raw/BPI Challenge 2017.xes.gz`
and is not included in the repository.

```sh
uv run python examples/reflex-builder/bpi2017/build.py \
  ".reflex-bpi2017/raw/BPI Challenge 2017.xes.gz" \
  .reflex-bpi2017/builder-run
```

The run writes a verified `lens.json`, `case-scores.csv`, and `summary.json`.
`case-scores.csv` contains held-out case IDs, completion times, feature
summaries, scores, cluster assignments, and source evidence IDs. Review the
highest-scoring cases against their source process histories before assigning
meaning to any flag.
