"""Input, observation, and action records written to the event ledger."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import NAMESPACE_URL, uuid4, uuid5

from pydantic import Field, field_validator

from .base import FrozenModel


class Event(FrozenModel):
    """Common immutable envelope shared by every event kind."""

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
    """An immutable event received from an external or synthetic source."""

    event_kind: Literal["native"] = "native"


class ObservationEvent(Event):
    """A Lens finding stored in the same ledger as its supporting events."""

    event_kind: Literal["observation"] = "observation"
    lens_id: str
    lens_version: str
    subject: str


class ActionEvent(Event):
    """A request to run a registered function under a named permission."""

    event_kind: Literal["action"] = "action"
    action_function: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    required_permission: str
    retry_limit: int = Field(default=0, ge=0, le=10)


class ActionResultEvent(Event):
    """An immutable record of an action attempt or its replay status."""

    event_kind: Literal["action"] = "action"
    action_event_id: str
    status: Literal["succeeded", "failed", "skipped_replay"]
    attempt: int = Field(ge=1)


def stable_id(*parts: str) -> str:
    """Return a repeatable ID for records that must remain stable on replay."""
    return str(uuid5(NAMESPACE_URL, "reflex:" + ":".join(parts)))
