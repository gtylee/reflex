"""Typed aggregates derived from event-time windows."""

from __future__ import annotations

from datetime import datetime

from .base import FrozenModel


class EventWindowFeatures(FrozenModel):
    """A feature vector derived for one subject over one half-open interval.

    ``supporting_event_ids`` lets a reviewer trace each aggregate back to its
    source rows. ``feature_values`` must use the same names and ordering as the
    model schema that consumes it. ``cohort`` is descriptive context for
    comparisons; it is not itself a prediction or causal label.
    """

    subject: str
    window_start: datetime
    window_end: datetime
    feature_values: dict[str, float]
    supporting_event_ids: tuple[str, ...]
    event_count: int
    cohort: str = "all"
