"""Generic training-only selection, calibration, and packaging for REFLEX lenses.

Dataset meaning and feature construction belong to adapters outside the core.
This module accepts normalized EventWindowFeatures and interchangeable
candidate implementations. It never accepts evaluator labels or source-specific
schemas.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .models.events import stable_id
from .models.features import EventWindowFeatures
from .models.lenses import Observation

PACKAGE_FORMAT = "reflex.lens-package.v1"


class LensBuilderCandidate(Protocol):
    """A domain-neutral candidate operating on normalized feature rows."""

    lens_id: str
    version: str
    feature_names: tuple[str, ...]
    metric_name: str

    def fit(self, rows: Sequence[EventWindowFeatures]) -> Any:
        """Fit model state using only the rows supplied."""

    def validation_loss(self, model: Any, rows: Sequence[EventWindowFeatures]) -> float:
        """Return a training-only selection metric where lower is better."""

    def score(self, model: Any, rows: Sequence[EventWindowFeatures]) -> Sequence[float]:
        """Return one anomaly score per row, in input order."""

    def explain(self, model: Any, row: EventWindowFeatures) -> Mapping[str, Any]:
        """Return JSON-compatible details for one score."""

    def encode(self, model: Any) -> Mapping[str, Any]:
        """Return JSON-compatible fitted parameters for the package."""

    def decode(self, payload: Mapping[str, Any]) -> Any:
        """Restore model state from a saved package."""


@dataclass
class FittedLensPackage:
    """A selected lens and its portable metadata for scoring new rows."""

    candidate: LensBuilderCandidate
    model: Any
    manifest: dict[str, Any]

    @property
    def lens_id(self) -> str:
        return str(self.manifest["lens_id"])

    @property
    def version(self) -> str:
        return str(self.manifest["lens_version"])

    @property
    def threshold(self) -> float:
        return float(self.manifest["threshold"])

    def score(self, rows: Sequence[EventWindowFeatures]) -> list[Observation]:
        """Score rows and return evidence-linked observations at the frozen cutoff."""
        expected_schema = tuple(self.manifest["feature_schema"])
        for row in rows:
            if tuple(row.feature_values) != expected_schema:
                raise ValueError("scoring rows do not match the saved lens feature schema")
            if row.window_start.tzinfo is None or row.window_end.tzinfo is None:
                raise ValueError("scoring windows must include a timezone")
            if row.window_end <= row.window_start:
                raise ValueError("scoring windows must have positive duration")
            if not all(math.isfinite(float(value)) for value in row.feature_values.values()):
                raise ValueError("scoring feature values must be finite numbers")
        values = _scores(self.candidate, self.model, rows)
        observations = []
        for row, score in zip(rows, values, strict=True):
            details = dict(self.candidate.explain(self.model, row))
            # Fail here, at the package boundary, instead of discovering that an
            # explanation cannot be persisted only when a host serializes it.
            _canonical_json(details)
            observations.append(Observation(
                observation_id=stable_id(self.lens_id, self.version, row.subject,
                                         row.window_start.isoformat(), row.window_end.isoformat()),
                observation_type="anomaly_score", lens_id=self.lens_id, lens_version=self.version,
                subject=row.subject, window_start=row.window_start, window_end=row.window_end,
                value={**details, "score": score, "threshold": self.threshold,
                       "is_anomaly": score > self.threshold},
                supporting_event_ids=row.supporting_event_ids,
                model_version=self.version,
                quality={"calibrated": True, "threshold_source": "calibration",
                         "training_cutoff": self.manifest["training_end"],
                         "selection_metric": self.manifest["selection_metric"],
                         "selection_loss": self.manifest["selection_loss"],
                         "calibration_rate": self.manifest["calibration_rate"]},
                provenance={"package_format": PACKAGE_FORMAT,
                            "package_sha256": self.manifest["package_sha256"],
                            "candidate_ref": self.manifest["candidate_ref"]},
            ))
        return observations

    def save(self, path: str | Path) -> Path:
        """Write a stable, standalone JSON package for later reload."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(_canonical_json(self.manifest) + "\n", encoding="utf-8")
        return target


def build_lens_package(
    *,
    fit_rows: Sequence[EventWindowFeatures],
    selection_rows: Sequence[EventWindowFeatures],
    calibration_rows: Sequence[EventWindowFeatures],
    candidates: Sequence[LensBuilderCandidate],
    goal: str,
    calibration_rate: float = 0.01,
    source_fingerprint: str | None = None,
) -> FittedLensPackage:
    """Select on a later training block, refit on training, then calibrate.

    This interface deliberately has no later evaluation rows or labels.
    """
    fit_rows = tuple(fit_rows)
    selection_rows = tuple(selection_rows)
    calibration_rows = tuple(calibration_rows)
    candidates = tuple(candidates)
    if not fit_rows or not selection_rows or not calibration_rows:
        raise ValueError("fit, selection, and calibration rows must each be non-empty")
    if not candidates:
        raise ValueError("at least one lens candidate is required")
    if not goal.strip():
        raise ValueError("goal must describe the intended review decision")
    if not 0 < calibration_rate < 0.5:
        raise ValueError("calibration_rate must be between 0 and 0.5")
    _validate_rows(fit_rows, selection_rows, calibration_rows)
    schema = tuple(fit_rows[0].feature_values)
    candidate_refs = [f"{candidate.lens_id}@{candidate.version}" for candidate in candidates]
    if len(set(candidate_refs)) != len(candidate_refs):
        raise ValueError("candidate IDs and versions must be unique")
    if len({candidate.metric_name for candidate in candidates}) != 1:
        raise ValueError("all candidates must report the same selection metric")

    # Compare candidates on the later training block only. Evaluator labels and
    # the eventual evaluation rows have no place in this reusable interface.
    comparisons = []
    for candidate in candidates:
        if tuple(candidate.feature_names) != schema:
            raise ValueError(
                f"{candidate.lens_id}@{candidate.version} feature schema does not match input rows"
            )
        fit_model = candidate.fit(fit_rows)
        loss = float(candidate.validation_loss(fit_model, selection_rows))
        if not math.isfinite(loss):
            raise ValueError(f"{candidate.lens_id}@{candidate.version} returned a non-finite metric")
        comparisons.append((loss, f"{candidate.lens_id}@{candidate.version}", candidate))
    selection_loss, _, selected = min(comparisons, key=lambda item: (item[0], item[1]))

    # Refit the winner with all training evidence after candidate choice.
    final_rows = fit_rows + selection_rows
    model = selected.fit(final_rows)

    # Calibration sees a frozen model and sets only its point-score cutoff.
    # Case grouping or workload budgets remain application-level policy.
    calibration_scores = _scores(selected, model, calibration_rows)
    threshold = float(np.quantile(calibration_scores, 1.0 - calibration_rate, method="higher"))
    calibration_alerts = sum(score > threshold for score in calibration_scores)
    encoded = dict(selected.encode(model))
    _canonical_json(encoded)
    candidate_ref = f"{selected.lens_id}@{selected.version}"
    manifest: dict[str, Any] = {
        "format": PACKAGE_FORMAT,
        "goal": goal.strip(),
        "lens_id": selected.lens_id,
        "candidate_ref": candidate_ref,
        "candidate_version": selected.version,
        "provider": getattr(selected, "provider", None),
        "provider_version": getattr(selected, "provider_version", None),
        "feature_schema": list(selected.feature_names),
        "selection_metric": selected.metric_name,
        "selection_loss": selection_loss,
        "candidate_selection": [
            {"candidate_ref": ref, "metric": candidate.metric_name, "loss": loss}
            for loss, ref, candidate in sorted(comparisons, key=lambda item: (item[0], item[1]))
        ],
        "threshold": threshold,
        "comparator": ">",
        "calibration_rate": calibration_rate,
        "calibration_alert_count": calibration_alerts,
        "calibration_row_count": len(calibration_rows),
        "fit_row_count": len(fit_rows),
        "selection_row_count": len(selection_rows),
        "refit_row_count": len(final_rows),
        "fit_data_sha256": _rows_digest(fit_rows),
        "selection_data_sha256": _rows_digest(selection_rows),
        "refit_data_sha256": _rows_digest(final_rows),
        "calibration_data_sha256": _rows_digest(calibration_rows),
        "fit_start": min(row.window_start for row in fit_rows).isoformat(),
        "fit_end": max(row.window_end for row in fit_rows).isoformat(),
        "selection_start": min(row.window_start for row in selection_rows).isoformat(),
        "selection_end": max(row.window_end for row in selection_rows).isoformat(),
        "training_start": min(row.window_start for row in final_rows).isoformat(),
        "training_end": max(row.window_end for row in final_rows).isoformat(),
        "calibration_start": min(row.window_start for row in calibration_rows).isoformat(),
        "calibration_end": max(row.window_end for row in calibration_rows).isoformat(),
        "source_fingerprint": source_fingerprint,
        "model": encoded,
    }
    package_sha256 = hashlib.sha256(_canonical_json(manifest).encode()).hexdigest()
    manifest["package_sha256"] = package_sha256
    manifest["lens_version"] = package_sha256[:16]
    return FittedLensPackage(selected, model, manifest)


def load_lens_package(path: str | Path, candidates: Sequence[LensBuilderCandidate]) -> FittedLensPackage:
    """Verify and restore a package against an explicitly supplied library.

    A package contains model data, not executable code. The caller controls
    which candidate providers are available to restore and run it.
    """
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("format") != PACKAGE_FORMAT:
        raise ValueError(f"unsupported lens package format: {manifest.get('format')!r}")
    expected_sha = manifest.get("package_sha256")
    expected_version = manifest.get("lens_version")
    unsigned = {key: value for key, value in manifest.items()
                if key not in {"package_sha256", "lens_version"}}
    actual_sha = hashlib.sha256(_canonical_json(unsigned).encode()).hexdigest()
    if expected_sha != actual_sha or expected_version != actual_sha[:16]:
        raise ValueError("lens package content does not match its recorded version")
    library = {}
    for candidate in candidates:
        ref = f"{candidate.lens_id}@{candidate.version}"
        if ref in library:
            raise ValueError(f"candidate library contains duplicate provider {ref!r}")
        library[ref] = candidate
    try:
        candidate = library[manifest["candidate_ref"]]
    except KeyError as error:
        raise KeyError(f"candidate library does not provide {manifest['candidate_ref']!r}") from error
    if list(candidate.feature_names) != manifest["feature_schema"]:
        raise ValueError("candidate feature schema differs from the saved lens package")
    model = candidate.decode(manifest["model"])
    return FittedLensPackage(candidate, model, manifest)


def _scores(candidate: LensBuilderCandidate, model: Any,
            rows: Sequence[EventWindowFeatures]) -> list[float]:
    values = list(candidate.score(model, rows))
    if len(values) != len(rows):
        raise ValueError(f"{candidate.lens_id}@{candidate.version} returned the wrong score count")
    scores = [float(value) for value in values]
    if not all(math.isfinite(value) for value in scores):
        raise ValueError(f"{candidate.lens_id}@{candidate.version} returned non-finite scores")
    return scores


def _validate_rows(*partitions: Sequence[EventWindowFeatures]) -> None:
    schemas = {tuple(row.feature_values) for partition in partitions for row in partition}
    if len(schemas) != 1:
        raise ValueError("all partitions must use the same ordered feature schema")
    if not next(iter(schemas)):
        raise ValueError("feature rows must contain at least one named numeric feature")
    event_ids = [event_id for partition in partitions for row in partition
                 for event_id in row.supporting_event_ids]
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("source evidence IDs must not be reused across partitions")
    latest_end: datetime | None = None
    for partition in partitions:
        for row in partition:
            if row.window_start.tzinfo is None or row.window_end.tzinfo is None:
                raise ValueError("feature windows must include a timezone")
            if row.window_end <= row.window_start:
                raise ValueError("feature windows must have positive duration")
            if row.event_count < 0:
                raise ValueError("event_count must be nonnegative")
            if not all(math.isfinite(float(value)) for value in row.feature_values.values()):
                raise ValueError("feature values must be finite numbers")
        earliest_end = min(row.window_end for row in partition)
        if latest_end is not None and earliest_end < latest_end:
            raise ValueError("fit, selection, and calibration partitions must be chronologically ordered")
        latest_end = max(row.window_end for row in partition)


def _rows_digest(rows: Sequence[EventWindowFeatures]) -> str:
    records = [{
        "subject": row.subject,
        "window_start": row.window_start.isoformat(),
        "window_end": row.window_end.isoformat(),
        "feature_values": dict(row.feature_values),
        "supporting_event_ids": list(row.supporting_event_ids),
        "event_count": row.event_count,
        "cohort": row.cohort,
    } for row in rows]
    return hashlib.sha256(_canonical_json(records).encode()).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False)


__all__ = ["PACKAGE_FORMAT", "LensBuilderCandidate", "FittedLensPackage",
           "build_lens_package", "load_lens_package"]
