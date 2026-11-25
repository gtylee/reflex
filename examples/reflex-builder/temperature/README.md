# Three office temperature cases per week

REFLEX opened 39 cases across 22 calendar weeks in the held-out NAB office temperature series. The busiest week had three cases. The budget removed 27 of 66 candidate cases (41%) and retained detection of the one labeled window. Nine queued cases opened inside that window and 30 opened elsewhere. This series measures workload well, but one broad incident window gives limited evidence about detection quality.

## Protocol

The goal was to surface unusually high or low hourly temperatures. The run used the clock printed in the source file, encoded as UTC, and retained the original sensor units. July through September 2013 supplied the initial fit. On the October training check, the pooled median had a median absolute error of 3.59 sensor units. Clustering comparable hours cut it to 3.26. REFLEX refitted that lens on all 2,477 July–October readings. [NAB source series](https://github.com/numenta/NAB/blob/ea702d75cc2258d9d7dd35ca8e5e2539d71f3140/data/realKnownCause/ambient_temperature_system_failure.csv)

The lens clusters the historical medians of weekday and weekend hours into three groups. A new timestamp selects its peer group before the lens reads the temperature. The fitted group medians were 68.86, 70.79 and 72.93. The score is the absolute difference from the group median divided by the group's robust scale, 1.4826 times its median absolute deviation.

The 1,464 November–December readings set a 95th-percentile cutoff of 2.704 through the shared REFLEX package builder. The source-specific grouping and weekly budget remain in this example's review policy. At that cutoff, calibration produced 15 candidate cases; nine would have entered the queue.

Evaluation started January 1, 2014 with an empty queue and ended May 28 at 15:00. Readings strictly above the score cutoff opened a case. Consecutive hourly flags in the same direction continued that case, including across midnight or a week boundary. A normal reading, gap or direction change ended it. On opening, the first three cases in a Monday–Sunday week entered the review queue. Further cases that week remained recorded as suppressed. The decision used information available when the case opened.

## What reached the reviewer

| Held-out measure | Result |
| --- | ---: |
| Readings / score crossings | 3,326 / 201 |
| Candidate / queued / suppressed cases | 66 / 39 / 27 |
| Weeks with 0 / 1 / 2 / 3 queued cases | 5 / 5 / 2 / 10 |
| Labeled windows detected, uncapped / capped | 1 of 1 / 1 of 1 |
| Queued openings inside / outside the window | 9 / 30 |

The published window ran from March 29 at 15:00 through April 20 at 22:00. It contained 363 observed hourly readings, with gaps in the series. The evaluator counted a detection when a case **opened** inside the inclusive window. The first such admission came ten hours after the window began. Five additional openings within the window were suppressed. The source provides a time window for comparison and gives no individual verdict on the 30 cases outside it. [NAB label windows](https://github.com/numenta/NAB/blob/ea702d75cc2258d9d7dd35ca8e5e2539d71f3140/labels/combined_windows.json)

| Reading in the warm peer group | Observed | Expected | Score | Decision |
| --- | ---: | ---: | ---: | --- |
| March 30 at 01:00 | 65.50 | 72.93 | 3.316 | First admission in the labeled window |
| January 23 at 13:00 | 72.90 | 72.93 | 0.011 | Below cutoff |

## Reproduce

With the pinned NAB checkout at `.reflex-nab/source`, run from the REFLEX repository root.

```sh
uv run python examples/reflex-builder/temperature/run.py \
  .reflex-nab/source/data/realKnownCause/ambient_temperature_system_failure.csv \
  .reflex-builder-temperature
uv run python examples/reflex-builder/temperature/evaluate.py \
  .reflex-builder-temperature .reflex-nab/source/labels/combined_windows.json
```

The first command writes the fitted lens package, a separate `review-policy.json`, all scores, and the queued or suppressed disposition of each candidate. It reloads the lens and verifies a representative source observation through the REFLEX runtime. The second command reads the frozen predictions, checks chronological admission and the cap for every week, and writes the weekly counts and window matches to `evaluation.json`.
