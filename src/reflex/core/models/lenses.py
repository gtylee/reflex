"""Lens definitions, observations, and candidate lifecycle records."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import Field, field_validator, model_validator

from .base import FrozenModel
from .events import ObservationEvent


class LensKind(StrEnum):
    SQL = "sql"
    AUTOENCODER = "autoencoder"
    ISOLATION_FOREST = "isolation_forest"


class LensVersion(FrozenModel):
    """A named, versioned method for identifying a pattern in events."""

    lens_id: str
    version: str
    kind: LensKind
    description: str
    code: str | None = None
    feature_schema: tuple[str, ...] = ()
    preprocessing: dict[str, Any] = Field(default_factory=dict)
    training_cutoff: datetime | None = None
    artifact_path: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class Observation(FrozenModel):
    """A finding with its subject, time window, method, and supporting evidence."""

    observation_id: str = Field(default_factory=lambda: str(uuid4()))
    observation_type: str
    lens_id: str
    lens_version: str
    subject: str
    window_start: datetime
    window_end: datetime
    value: Any
    supporting_event_ids: tuple[str, ...] = ()
    state_snapshot: dict[str, Any] = Field(default_factory=dict)
    model_version: str | None = None
    quality: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("window_start", "window_end")
    @classmethod
    def require_window_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("observation windows must include a timezone")
        return value.astimezone(timezone.utc)

    def as_event(self) -> ObservationEvent:
        """Convert the finding into the ledger's dedicated observation event."""
        return ObservationEvent(
            event_id=self.observation_id,
            event_type=f"observation.{self.observation_type}",
            event_time=self.window_end,
            entity_ids=(self.subject,),
            source=f"lens:{self.lens_id}@{self.lens_version}",
            provenance={**dict(self.provenance), "model_version": self.model_version},
            parent_event_ids=self.supporting_event_ids,
            causation_id=self.supporting_event_ids[-1] if self.supporting_event_ids else None,
            schema_ref="reflex.observation@1.0",
            payload={
                "value": self.value,
                "window_start": self.window_start.isoformat(),
                "window_end": self.window_end.isoformat(),
                "state_snapshot": dict(self.state_snapshot),
                "quality": dict(self.quality),
            },
            lens_id=self.lens_id,
            lens_version=self.lens_version,
            subject=self.subject,
        )


class CandidateStatus(StrEnum):
    PROPOSED = "proposed"
    TESTING = "testing"
    VALIDATED = "validated"
    APPROVED = "approved"
    ACTIVE = "active"
    RETIRED = "retired"
    REJECTED = "rejected"


class LensCandidate(FrozenModel):
    """A proposed Lens with explicit time boundaries and validation results."""

    candidate_id: str = Field(default_factory=lambda: str(uuid4()))
    lens_id: str
    version: str
    sql: str
    description: str
    supporting_observations: tuple[str, ...] = ()
    applicability: str
    expected_outputs: dict[str, Any] = Field(default_factory=dict)
    training_boundary: datetime
    validation_boundary: datetime
    estimated_cost_ms: float = Field(default=0.0, ge=0.0)
    limitations: tuple[str, ...] = ()
    status: CandidateStatus = CandidateStatus.PROPOSED
    validation: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_boundaries(self) -> "LensCandidate":
        if self.training_boundary.tzinfo is None or self.validation_boundary.tzinfo is None:
            raise ValueError("candidate boundaries must include a timezone")
        if self.validation_boundary <= self.training_boundary:
            raise ValueError("validation boundary must follow the training boundary")
        return self
