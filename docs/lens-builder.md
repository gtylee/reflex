# Training and packaging a REFLEX lens

`reflex.core` provides a dataset-neutral builder for selecting a scoring method,
calibrating its threshold, and saving a versioned package. Dataset parsing,
feature meaning, candidate implementations, review policies, labels, and
experiments belong in application code such as `examples/reflex-builder/`.
The core does not import those examples.

## Input contract

An adapter converts source records to `EventWindowFeatures`:

```python
EventWindowFeatures(
    subject="account-42",
    window_start=...,                 # timezone-aware
    window_end=...,                   # timezone-aware, after window_start
    feature_values={"amount": 12.5}, # ordered numeric schema
    supporting_event_ids=("src:17",),
    event_count=1,
    cohort="all",
)
```

The row contains normalized numbers and provenance. The core does not know
that `amount` means money, that a `subject` is an account, or how records were
read. Evidence IDs cannot be reused across fit, selection, and calibration
partitions. Those partitions must progress forward by window end.

## Candidate contract

An application supplies one or more candidate objects that implement
`reflex.core.LensBuilderCandidate`. Each candidate declares its feature names and
version, fits model state, reports a training-only selection loss (lower is
better), scores rows, explains an individual score, and encodes/decodes its
fitted state. It can use clustering, a robust baseline, a tree model, or another
method without adding dataset-specific code to `reflex.core`.

Candidate comparisons must use the same metric and units. For example, taxi
and temperature candidates compare median absolute prediction error in their
own source units. The BPI process example compares median normalized
reconstruction error for both its pooled and clustered candidates.
When a candidate's behavior changes, its version must change too; package
loading resolves the exact `lens_id@version` reference from the supplied code.

## Build flow

```python
package = build_lens_package(
    fit_rows=earlier_training_rows,
    selection_rows=later_training_rows,
    calibration_rows=held_back_calibration_rows,
    candidates=application_candidates,
    goal="Review unusually long completed cases",
    calibration_rate=0.02,
    source_fingerprint="sha256-of-source-file",
)
package.save("lens.json")
```

Candidate comparison uses only the fit and selection partitions. The selected
candidate is refit with both training partitions, then the threshold is set to
the requested upper-tail quantile of calibration scores. There is no evaluation
data or label argument in this API. If the review goal counts incidents or
weekly work rather than individual points, keep that grouping and budget policy
in the application layer; a point threshold and a queue cap are separate
decisions.

The package records its feature schema, candidate/provider reference,
selection metric and results, threshold, calibration count, evidence digests,
time boundaries, source fingerprint, and JSON model parameters. A SHA-256
digest versions the package and detects edits. Loading requires the application
to supply the candidate implementation library:

```python
package = load_lens_package("lens.json", candidates=application_candidates)
observations = package.score(new_rows)
```

The package contains portable fitted data, not executable code. This keeps
candidate code independently installable and makes the trust boundary explicit.
`FittedLensPackage.score` returns evidence-linked `Observation` records with a
frozen cutoff; an application can adapt those records to a Reaction or another
host interface.

## Worked examples

- [NYC taxi demand](../examples/reflex-builder/taxi/README.md): low-demand
  half-hours and weekday/time-slot peers.
- [Office temperature](../examples/reflex-builder/temperature/README.md):
  two-sided deviations with a separate weekly case budget.
- [BPI Challenge 2017](../examples/reflex-builder/bpi2017/README.md):
  completed application cases, outcome activities removed from features, and
  pooled-versus-clustered process profiles.

These examples own their data adapters, feature engineering, interpretation,
and evaluation. Reusable event and package contracts stay under `src/reflex/core`.
