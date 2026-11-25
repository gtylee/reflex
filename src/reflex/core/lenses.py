"""Deterministic, versioned SQL lenses that produce evidence-linked findings."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable

from .models.events import stable_id
from .models.lenses import LensKind, LensVersion, Observation
from .storage import EventStore


class SqlLens:
    """A named SQL query that turns matching event rows into observations.

    SQL is stored as the lens definition so results carry the exact query and
    version in their provenance. Queries execute through ``EventStore``'s
    bounded local adapter.
    """

    def __init__(self, lens_id: str, version: str, description: str, sql: str):
        """Create an immutable lens definition for the supplied query text."""
        self.definition = LensVersion(lens_id=lens_id, version=version, kind=LensKind.SQL,
                                      description=description, code=sql)

    def replay(self, store: EventStore, *, start: datetime | None = None, end: datetime | None = None,
               state_snapshot: dict[str, Any] | Callable[[dict[str, Any]], dict[str, Any]] | None = None,
               limit: int = 1000) -> list[Observation]:
        """Evaluate the query and make one point-in-time finding per result row.

        The SQL result must include ``event_id``, ``event_type``,
        ``event_time``, ``entity_ids``, and ``attributes``. Optional time bounds
        are half-open through the store adapter. A state callback can attach a
        per-row snapshot without changing the event evidence references.
        """
        rows = store.query_sql_lens(self.definition.code or "", limit=limit, start=start, end=end)
        result = []
        for row in rows:
            entity_ids = row.get("entity_ids", [])
            if isinstance(entity_ids, str):
                entity_ids = json.loads(entity_ids)
            attributes = row.get("attributes", {})
            if isinstance(attributes, str):
                attributes = json.loads(attributes)
            timestamp = row["event_time"]
            result.append(Observation(
                observation_id=stable_id(self.definition.lens_id, self.definition.version, str(row["event_id"])),
                observation_type="sql_threshold_breach", lens_id=self.definition.lens_id,
                lens_version=self.definition.version,
                subject=str(entity_ids[0] if entity_ids else "unassigned"),
                window_start=timestamp, window_end=timestamp,
                value={"event_type": row["event_type"], "attributes": attributes},
                supporting_event_ids=(str(row["event_id"]),),
                state_snapshot=(state_snapshot(row) if callable(state_snapshot) else state_snapshot) or {},
                quality={"deterministic": True, "replayable": True},
                provenance={"sql": self.definition.code, "source": "duckdb_event_ledger"},
            ))
        return result


def __getattr__(name: str):
    """Expose the old transaction example query only for compatibility imports."""
    if name != "DEFAULT_SQL":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from reflex.experiments.transaction_lenses import TRANSACTION_HIGH_AMOUNT_SQL

    return TRANSACTION_HIGH_AMOUNT_SQL


__all__ = ["SqlLens", "DEFAULT_SQL"]
