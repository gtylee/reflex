"""Deterministic, versioned SQL lenses."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable

from .models.events import stable_id
from .models.lenses import LensKind, LensVersion, Observation
from .storage import EventStore


DEFAULT_SQL = """
SELECT event_id, event_type, event_time, entity_ids, attributes
FROM events
WHERE TRY_CAST(json_extract_string(attributes, '$.amount') AS DOUBLE) >= 4000
  AND event_kind = 'native'
""".strip()


class SqlLens:
    def __init__(self, lens_id: str, version: str, description: str, sql: str):
        self.definition = LensVersion(lens_id=lens_id, version=version, kind=LensKind.SQL,
                                      description=description, code=sql)

    def replay(self, store: EventStore, *, start: datetime | None = None, end: datetime | None = None,
               state_snapshot: dict[str, Any] | Callable[[dict[str, Any]], dict[str, Any]] | None = None,
               limit: int = 1000) -> list[Observation]:
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
