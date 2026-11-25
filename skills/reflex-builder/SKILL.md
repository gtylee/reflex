---
name: reflex-builder
description: Build a reviewable REFLEX lens and Reaction from a training dataset, operating context and a detection goal. Use for creating an evidence-linked detector, fitting peer groups, calibrating alert volume and investigating anomalous examples.
---

# Build a REFLEX

Translate the user's goal into one decision a reviewer can make. Record what an observation represents, when the decision is made, the direction of interest, the unit and the review budget. If the goal is broad, propose one measurable interpretation and disclose it. If a requested outcome needs a field the source does not contain, clarify the gap before treating a proxy as the target.

## Fit the comparison

Inspect source records, timestamps, gaps, repeated IDs, missing fields, units and group sizes. Keep raw IDs for evidence. Write down the information available at scoring time. Store future outcomes and anomaly labels in the evaluator. For time-ordered data, split by time, keeping simultaneous observations together and feature windows on the correct side of the boundary. If only training data is available, reserve a later portion for calibration when feasible. Describe any remaining evaluation gap.

Build a simple baseline in the units of the goal. Examples include a median and robust scale for a measurement, the frequency of a category, or elapsed time between two process steps. Choose features from the goal and the training profile. Use one or more later blocks *inside training* to compare candidate baselines. For unlabeled data report what the proxy measures, such as median absolute prediction error. A lower prediction error alone cannot establish detection precision.

When working in the REFLEX repository, normalize rows to `EventWindowFeatures` and use `reflex.core.build_lens_package` for training-only candidate comparison, refitting, calibration, and package creation. Implement each candidate as its own type with a declared feature schema, selection metric, score, explanation, and encode/decode methods. Supply the candidate list explicitly; load a saved package with the matching candidate library. Do not put dataset readers, feature meaning, labels, evaluation code, or example-specific policies in `reflex.core`.

Where peers matter, cluster training profiles using features a new observation can be assigned from before seeing the outcome being scored. Inspect cluster counts, typical values, within-group spread and concrete members. Compare a peer lens with the simple baseline. Use a pooled fallback for unsupported groups. Handle missing observations and zero spread explicitly. Persist the mapping, preprocessing, scale and fallback. For data with no defensible segmentation, retain the pooled lens and report why.

## Calibrate and explain

Freeze the selected score, then apply it to a later calibration slice to set the cutoff. Record the comparator, ties, actual alert count, denominator and unit of the review budget. A point-level cutoff and an incident-level budget require different counting. Keep the evaluation slice out of design choices. When training support or calibration resolution is too small for the requested tail rate, report that limit.

Score later observations with the frozen artifact. Link each flag to the source ID and show the observed value, expected value, peer group, score, threshold and plain-language reason. Group related flags for review using a stated rule, such as consecutive misses in the same direction on one series. Preserve individual evidence and count both flags and cases. Show a comparable unflagged observation. If labels are available, open them after predictions are saved and state the matching unit and denominator.

## Deliver a usable Reaction

Place fitted parameters and the feature schema in a versioned artifact. Record the source fingerprint, time boundaries, objective, calibration result and method version. The core package stores fitted data and candidate references, not executable application code. Provide a callable that takes a REFLEX Event and emits an ObservationEvent with stable identity, parent evidence and the review flag. Use a FunctionSpec and a ReactionDefinition so a host can load the lens through its FunctionRegistry. Keep dataset mappings and examples outside the reusable core. Exercise a saved lens on a source event through the runtime and show the result.

For worked examples in the REFLEX repository, see `examples/reflex-builder/`. Taxi and temperature show event-level Reactions; BPI 2017 shows a case-level process lens. Their source mappings, feature definitions, candidate classes, thresholds and policies belong to those examples. The shared package contract is described in `docs/lens-builder.md`.
