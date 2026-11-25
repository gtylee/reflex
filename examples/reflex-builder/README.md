# REFLEX lens-builder examples

These are application-owned demonstrations built on the shared API in
[`reflex.core`](../../docs/lens-builder.md). Each directory owns its source
adapter, candidate lens types, goal, time splits, review policy, and evaluator.
The core has no imports from this directory and contains no taxi, temperature,
or BPI-specific features.

| Example | Review decision | Domain-specific behavior |
| --- | --- | --- |
| [Taxi](taxi/README.md) | Which half-hours are unusually quiet? | Weekday/time-slot peer clusters |
| [Temperature](temperature/README.md) | Which readings are unusually high or low? | Direction-aware case grouping and weekly queue cap |
| [BPI 2017](bpi2017/README.md) | Which completed process cases differ from historical profiles? | Case aggregation, outcome-event exclusion, process clusters |

Each builder passes normalized `EventWindowFeatures` plus an explicit list of
candidate implementations to `build_lens_package`. Saved packages contain
fitted state and metadata, while candidate code remains in the example or its
separately installed provider. The held-out evaluation data and labels are
handled by the example scripts after the package has been frozen.
