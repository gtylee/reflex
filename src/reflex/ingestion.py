"""Provider-neutral boundary for later public or licensed news adapters."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping, Protocol, runtime_checkable

from reflex.core.models.events import NativeEvent, stable_id


@runtime_checkable
class NewsStoryAdapter(Protocol):
    """Normalize one source record into the common REFLEX native-event shape."""

    source_id: str

    def to_event(self, record: Mapping[str, Any]) -> NativeEvent: ...


class GenericNewsAdapter:
    """Adapter for an already-normalized news record; no source SDK is required."""

    def __init__(self, source_id: str):
        if not source_id.strip():
            raise ValueError("source_id is required")
        self.source_id = source_id

    def to_event(self, record: Mapping[str, Any]) -> NativeEvent:
        story_id = str(record.get("story_id") or record.get("source_story_id") or "").strip()
        headline = str(record.get("headline") or "").strip()
        published_at = record.get("published_at")
        if not story_id or not headline or published_at is None:
            raise ValueError("news record requires story_id, headline, and published_at")
        if isinstance(published_at, str):
            published_at = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        revision = int(record.get("revision", 0))
        source_record_id = str(record.get("source_record_id") or story_id)
        companies = [str(value) for value in record.get("companies", [])]
        themes = [str(value) for value in record.get("themes", [])]
        payload = {
            "story_id": story_id, "revision": revision, "headline": headline,
            "companies": companies, "themes": themes,
            "relationships": list(record.get("relationships", [])),
            "candidate_relationships": list(record.get("candidate_relationships", [])),
        }
        return NativeEvent(
            event_id=stable_id("news-source", self.source_id, source_record_id, str(revision)),
            event_type="news.story", event_time=published_at,
            entity_ids=tuple(companies), source=f"news-source:{self.source_id}",
            provenance={"source_id": self.source_id, "source_record_id": source_record_id,
                        "source_url": record.get("source_url"), "ingestion_adapter": "generic-news@1.0"},
            relationships=tuple(str(value) for value in record.get("relationship_refs", [])),
            correlation_id=f"news/{self.source_id}/{story_id}",
            idempotency_key=f"{self.source_id}:{source_record_id}:{revision}",
            schema_ref="portfolio.news-story@1.0", payload=payload, attributes=payload,
        )


__all__ = ["NewsStoryAdapter", "GenericNewsAdapter"]
