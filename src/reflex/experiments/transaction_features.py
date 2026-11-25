"""Transaction-domain event features used by selected REFLEX experiments.

The fields below assume transaction-like attributes such as amount, channel,
region, and latency. Keep new domain schemas in their own experiment profile
instead of adding them to the reusable core.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from math import log2
from typing import Iterable

import numpy as np

from reflex.core.models.events import Event
from reflex.core.models.features import EventWindowFeatures


# Keep the model input order explicit. Artifact manifests record these names so
# feature reordering is visible when a model is trained or loaded.
TRANSACTION_FEATURE_NAMES = (
    "event_count", "mean_amount", "std_amount", "max_amount", "high_amount_share",
    "mean_interarrival_hours", "std_interarrival_hours", "event_type_entropy",
    "event_type_diversity", "transition_count", "approval_share", "review_share",
    "manual_channel_share", "partner_channel_share", "mean_latency_minutes",
    "after_hours_share", "amount_per_event", "peer_event_count_ratio",
    "manual_approval_share", "partner_mean_amount_ratio", "amount_latency_correlation",
)


def transaction_event_features(events: Iterable[Event]) -> list[dict[str, float | str]]:
    """Return simple transaction-oriented values in stable event-time order.

    Interarrival time is measured separately for each event's first entity ID;
    events without an entity use ``unassigned``. This helper is event-level,
    while ``build_transaction_windows`` creates the aggregated model rows.
    """
    ordered = sorted(events, key=lambda item: (item.event_time, item.event_id))
    previous: dict[str, datetime] = {}
    rows = []
    for event in ordered:
        entity = event.entity_ids[0] if event.entity_ids else "unassigned"
        prior = previous.get(entity)
        gap = max(0.0, (event.event_time - prior).total_seconds() / 3600) if prior else 0.0
        previous[entity] = event.event_time
        rows.append({
            "event_id": event.event_id, "entity_id": entity,
            "interarrival_hours": gap,
            "amount": float(event.attributes.get("amount", 0.0) or 0.0),
            "latency_minutes": float(event.attributes.get("latency_minutes", 0.0) or 0.0),
            "is_after_hours": float(event.event_time.hour < 7 or event.event_time.hour >= 21),
        })
    return rows


def build_transaction_windows(
    events: Iterable[Event], *, start: datetime, end: datetime, window_days: int = 5,
) -> list[EventWindowFeatures]:
    """Aggregate transaction events into half-open entity windows.

    Only events in ``[start, end)`` are considered. Every event in a window is
    assigned to each of its entity IDs, or to ``unassigned`` when it has none.
    Empty entity windows are omitted. Event IDs are retained so each aggregate
    can be traced to its source records.

    Several domain fields have explicit defaults: missing numeric amounts and
    latencies are treated as zero, and missing channel/region values use
    ``unknown``. Callers should adapt these choices when zero does not mean
    "not measured" for their source.
    """
    if start.tzinfo is None or end.tzinfo is None or end <= start or window_days < 1:
        raise ValueError("start/end must be aware datetimes and window_days positive")
    ordered = sorted((event for event in events if start <= event.event_time < end), key=lambda e: (e.event_time, e.event_id))
    windows: list[EventWindowFeatures] = []
    cursor = start
    while cursor < end:
        window_end = min(cursor + timedelta(days=window_days), end)
        grouped: dict[str, list[Event]] = defaultdict(list)
        for event in ordered:
            if cursor <= event.event_time < window_end:
                for entity in event.entity_ids or ("unassigned",):
                    grouped[entity].append(event)
        counts_by_region: Counter[str] = Counter()
        entity_region: dict[str, str] = {}
        for entity, entity_events in grouped.items():
            # Peer volume is compared within the first event's region for this
            # subject-window. The cohort/region mapping is descriptive input,
            # not a causal adjustment.
            region = str(entity_events[0].attributes.get("region", "unknown"))
            entity_region[entity] = region
            counts_by_region[region] += len(entity_events)
        region_entities = Counter(entity_region.values())
        for entity, entity_events in grouped.items():
            amounts = np.asarray([float(e.attributes.get("amount", 0) or 0) for e in entity_events], dtype=float)
            latencies = np.asarray([float(e.attributes.get("latency_minutes", 0) or 0) for e in entity_events], dtype=float)
            times = [e.event_time for e in entity_events]
            gaps = np.asarray([max(0.0, (right - left).total_seconds() / 3600) for left, right in zip(times, times[1:])], dtype=float)
            type_counts = Counter(e.event_type for e in entity_events)
            probabilities = [value / len(entity_events) for value in type_counts.values()]
            entropy = -sum(p * log2(p) for p in probabilities if p)
            channels = Counter(str(e.attributes.get("channel", "unknown")) for e in entity_events)
            partner_amounts = [float(e.attributes.get("amount", 0) or 0) for e in entity_events
                               if e.attributes.get("channel") == "partner_api"]
            if len(amounts) > 1 and float(np.std(amounts)) > 0 and float(np.std(latencies)) > 0:
                amount_latency_correlation = float(np.corrcoef(amounts, latencies)[0, 1])
            else:
                amount_latency_correlation = 0.0
            high_amount = sum(amount > 1500 for amount in amounts)
            transitions = sum(left.event_type != right.event_type for left, right in zip(entity_events, entity_events[1:]))
            region = entity_region[entity]
            region_average = counts_by_region[region] / max(region_entities[region], 1)
            peer_ratio = len(entity_events) / max(1.0, region_average)
            values = {
                "event_count": float(len(entity_events)),
                "mean_amount": float(np.mean(amounts)), "std_amount": float(np.std(amounts)),
                "max_amount": float(np.max(amounts)), "high_amount_share": float(high_amount / len(amounts)),
                "mean_interarrival_hours": float(np.mean(gaps)) if len(gaps) else 0.0,
                "std_interarrival_hours": float(np.std(gaps)) if len(gaps) else 0.0,
                "event_type_entropy": float(entropy), "event_type_diversity": float(len(type_counts)),
                "transition_count": float(transitions),
                "approval_share": float(type_counts["approve"] / len(entity_events)),
                "review_share": float(type_counts["review"] / len(entity_events)),
                "manual_channel_share": float(channels["manual_override"] / len(entity_events)),
                "partner_channel_share": float(channels["partner_api"] / len(entity_events)),
                "mean_latency_minutes": float(np.mean(latencies)),
                "after_hours_share": float(np.mean([e.event_time.hour < 7 or e.event_time.hour >= 21 for e in entity_events])),
                "amount_per_event": float(np.sum(amounts) / len(amounts)),
                "peer_event_count_ratio": float(peer_ratio),
                "manual_approval_share": float(sum(e.event_type == "approve" and e.attributes.get("channel") == "manual_override" for e in entity_events) / len(entity_events)),
                "partner_mean_amount_ratio": float(np.mean(partner_amounts) / max(float(np.mean(amounts)), 1e-9)) if partner_amounts else 0.0,
                "amount_latency_correlation": amount_latency_correlation,
            }
            windows.append(EventWindowFeatures(
                subject=entity, window_start=cursor, window_end=window_end,
                feature_values=values, supporting_event_ids=tuple(e.event_id for e in entity_events),
                event_count=len(entity_events), cohort=str(entity_events[0].attributes.get("cohort", "all")),
            ))
        cursor = window_end
    return windows


# Historical names retained for notebooks that imported the old experiment
# helpers from ``reflex.core.features``. New callers should use the explicit
# transaction-domain names in this module.
FEATURE_NAMES = TRANSACTION_FEATURE_NAMES
event_level_features = transaction_event_features
build_entity_windows = build_transaction_windows

__all__ = [
    "TRANSACTION_FEATURE_NAMES", "transaction_event_features", "build_transaction_windows",
    "FEATURE_NAMES", "event_level_features", "build_entity_windows",
]
