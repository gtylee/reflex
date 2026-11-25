"""Typed aggregates derived from event-time windows."""

from __future__ import annotations

from datetime import datetime

from .base import FrozenModel


class EventWindowFeatures(FrozenModel):
    """A feature summary for one subject over one bounded time window."""

    subject: str
    window_start: datetime
    window_end: datetime
    feature_values: dict[str, float]
    supporting_event_ids: tuple[str, ...]
    event_count: int
    cohort: str = "all"
