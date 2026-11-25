# REFLEX Builder pilot on NYC taxi demand

The goal was to send unexpectedly quiet half-hours to a reviewer. The input was one passenger count per half-hour from the NAB NYC taxi series. I chose a target of 1% of half-hours during calibration. Timestamps preserve the clock shown in the source file. [Dataset at the study revision](https://github.com/numenta/NAB/blob/ea702d75cc2258d9d7dd35ca8e5e2539d71f3140/data/realKnownCause/nyc_taxi.csv)

## What the builder learned

July and August supplied a pooled passenger median and a competing peer model. The peer model took each weekday and half-hour combination, computed its median demand, then clustered those medians into four groups. A future Tuesday at 07:30 gets its group from the fitted timetable, before its actual passenger count is read.

On the September training check, the pooled median missed the count by 4,186.5 passengers at the median. The peer model missed by 1,659.5. REFLEX therefore refitted the peer model on all 4,416 July–September half-hours. The fitted medians, from quietest group to busiest, were 4,180, 12,913, 17,393 and 23,098.

The lens divides the shortfall below the peer median by that group's robust scale (1.4826 times its median absolute deviation, floored at one passenger). Counts above the median score zero. October supplied the 99th percentile cutoff of 1.735. A strict comparison flagged 14 of 1,488 calibration half-hours (0.94%).

## What a reviewer sees

| Source time | Passengers | Peer median | Score | Result |
| --- | ---: | ---: | ---: | --- |
| Jan 27, 2015 at 07:30 | 332 | 17,393 | 9.886 | Flag |
| Nov 3, 2014 at 17:00 | 17,392 | 17,393 | 0.001 | Below cutoff |

From November through January, **500 of 4,416 half-hours (11.3%)** crossed the cutoff. Grouping adjacent flagged half-hours produced **126 review cases**. Five published NAB anomaly windows intersected this period, including one that began in October. Each contained a flag. Forty cases overlapped these windows and **86 fell outside**. These are inclusive window-overlap counts. NAB supplies no individual verdict for the 86 cases. The workload exceeded the calibration target on this series.

## Reproduce the pilot

From the REFLEX repository, with the NAB checkout at `.reflex-nab/source`, run the following commands. The first builds the artifact, scores the evaluation period, and replays the highest-scoring record through the REFLEX runtime. The second compares the saved predictions with labels.

```sh
uv run python examples/reflex-builder/taxi/build.py \
  .reflex-nab/source/data/realKnownCause/nyc_taxi.csv .reflex-builder-pilot
uv run python examples/reflex-builder/taxi/evaluate.py \
  .reflex-builder-pilot .reflex-nab/source/labels/combined_windows.json
```

Open `lens.json` for the slot map and fitted baselines, `scores.csv` for all decisions, `cases.json` for grouped source IDs, and `replay.json` for the emitted ObservationEvent. The reusable workflow lives in [the REFLEX Builder skill](../../../skills/reflex-builder/SKILL.md).
