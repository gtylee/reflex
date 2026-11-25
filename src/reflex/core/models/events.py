"""Input, observation, and action records written to the event ledger."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid4, uuid5

from pydantic import Field, field_validator

from .base import FrozenModel


class Event(FrozenModel):
    """Common immutable envelope for source facts and derived records.

    ``event_type`` names the domain activity, while ``event_kind`` identifies
    the REFLEX record role (native, observation, or action). Keep source facts
    in ``attributes`` and normalized REFLEX metadata in the named lineage
    fields. ``payload`` carries the record-specific body for derived events.

    Timestamps are normalized to UTC at validation time. ``entity_ids`` may
    contain more than one participant, and an empty tuple is allowed when a
    source has no reliable entity key.
    """

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: str
    event_time: datetime
    entity_ids: tuple[str, ...] = ()
    attributes: dict[str, Any] = Field(default_factory=dict)
    source: str = "unknown"
    provenance: dict[str, Any] = Field(default_factory=dict)
    relationships: tuple[str, ...] = ()
    # Keep the discriminator optional for compatibility with older stored rows.
    event_kind: Literal["native", "observation", "action"] = "native"
    payload: dict[str, Any] = Field(default_factory=dict)
    parent_event_ids: tuple[str, ...] = ()
    correlation_id: str | None = None
    causation_id: str | None = None
    schema_ref: str | None = None
    idempotency_key: str | None = None
    access_scope: tuple[str, ...] = ()

    @field_validator("event_time")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("event_time must include a timezone")
        return value.astimezone(timezone.utc)

    def __eq__(self, other: Any) -> bool:
        # Keep legacy Event comparisons working when the ledger returns NativeEvent.
        if isinstance(other, Event):
            return self.model_dump(mode="python") == other.model_dump(mode="python")
        return NotImplemented


class NativeEvent(Event):
    """An immutable source fact, such as an imported log or transaction.

    Native events are the normal input to feature building and model fitting.
    Derived observations and action records share the ledger but should not be
    fed back as source data unless a workflow deliberately asks for them.
    """

    event_kind: Literal["native"] = "native"


class ObservationEvent(Event):
    """A Lens finding represented in the event ledger.

    ``parent_event_ids`` point to the input evidence. The event uses the end of
    the observation window as its event time, while the full interval and score
    remain in ``payload``.
    """

    event_kind: Literal["observation"] = "observation"
    lens_id: str
    lens_version: str
    subject: str


class ActionEvent(Event):
    """A request for the runtime to call a registered function.

    This record expresses intent; it does not execute code by itself. The
    runtime is responsible for resolving ``action_function`` through its
    allowlist, checking ``required_permission``, and applying retry and
    idempotency rules before any external side effect.
    """

    event_kind: Literal["action"] = "action"
    action_function: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    required_permission: str
    retry_limit: int = Field(default=0, ge=0, le=10)


class ActionResultEvent(Event):
    """An immutable record of one action attempt or replay decision."""

    event_kind: Literal["action"] = "action"
    action_event_id: str
    status: Literal["succeeded", "failed", "skipped_replay"]
    attempt: int = Field(ge=1)


def stable_id(*parts: str) -> str:
    """Return a deterministic UUID5 for a logical record identified by parts.

    Use this when replaying the same lens/window or source record should resolve
    to the same event ID. Include the lens version or other version boundary in
    ``parts`` when a changed definition should produce a distinct record.
    """
    return str(uuid5(NAMESPACE_URL, "reflex:" + ":".join(parts)))
