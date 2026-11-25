"""Small, deterministic news, operations, state, and curation experiments.

These are research harnesses. They emit synthetic events and simulated outcomes;
they never contact an external portfolio, news, or operations system.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import numpy as np

from reflex.core.models.events import Event, stable_id


@dataclass(frozen=True)
class PortfolioExposure:
    portfolio_id: str
    exposures: dict[str, float]


def synthetic_news_story(seed: int = 11) -> tuple[Event, ...]:
    """A developing rare-earth export story with duplicates and corrections."""
    start = datetime(2025, 2, 1, 8, tzinfo=timezone.utc)
    updates = (
        (0, "news.initial", "wire-a", "unconfirmed", "China may restrict rare-earth exports"),
        (5, "news.duplicate", "wire-b", "unconfirmed", "China may restrict rare-earth exports"),
        (35, "news.update", "wire-a", "expanded", "Restrictions may cover processed minerals"),
        (52, "news.contradiction", "wire-c", "disputed", "Officials deny a final decision"),
        (110, "news.official_confirmation", "official", "confirmed", "Export licensing restrictions announced"),
        (190, "news.correction", "wire-a", "corrected", "Draft scope excludes two mineral grades"),
        (245, "news.retraction", "wire-b", "retracted", "Earlier duplicate attributed the measure to the wrong ministry"),
    )
    result = []
    for revision, (minute, kind, source, status, text) in enumerate(updates):
        event_id = stable_id("news-story", str(seed), str(revision))
        result.append(Event(
            event_id=event_id, event_type=kind, event_time=start + timedelta(minutes=minute),
            entity_ids=("country:CN", "commodity:rare-earths"), source=source,
            attributes={"story_id": "story-rare-earth-001", "revision": revision,
                       "status": status, "topics": ["export_restriction", "rare_earths"],
                       "entities": ["China", "rare earths"], "text": text},
            provenance={"generator": "synthetic-news-v1", "seed": seed},
            relationships=("story-rare-earth-001",),
        ))
    return tuple(result)


def process_news_events(events: tuple[Event, ...], portfolio: PortfolioExposure,
                        state: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Route material new information and keep story/feedback state across batches."""
    memory = dict(state or {})
    seen_revisions = set(memory.get("seen_revisions", []))
    seen_content = set(memory.get("seen_content", []))
    reactions: list[dict[str, Any]] = []
    exposure = float(memory.get("rare_earths_exposure", portfolio.exposures.get("rare_earths", 0.0)))
    exposure += 0.40 * float(portfolio.exposures.get("materials_equity", 0.0))
    exposure += 0.25 * float(portfolio.exposures.get("industrial_metals", 0.0))
    threshold = float(memory.get("notification_threshold", 0.15))
    last_status = memory.get("story_status", "unknown")
    for event in sorted(events, key=lambda item: (item.event_time, item.event_id)):
        story_id = str(event.attributes["story_id"])
        revision = int(event.attributes["revision"])
        key = f"{story_id}:{revision}"
        if key in seen_revisions:
            reactions.append({"event_id": event.event_id, "action": "duplicate_suppressed", "score": 0.0})
            continue
        seen_revisions.add(key)
        status = str(event.attributes["status"])
        content_key = " ".join(str(event.attributes.get("text", "")).lower().split())
        if content_key and content_key in seen_content:
            reactions.append({"event_id": event.event_id, "action": "duplicate_suppressed", "score": 0.0,
                              "story_id": story_id, "revision": revision})
            continue
        if content_key:
            seen_content.add(content_key)
        topics = set(event.attributes.get("topics", []))
        score = min(1.0, exposure) * (1.0 if "rare_earths" in topics else 0.0)
        materially_changed = status in {"expanded", "confirmed", "corrected", "retracted"} and status != last_status
        if status == "retracted":
            action = "reopen_and_reassess"
            score = 0.0
        elif score > 0 and memory.get("investigation_requested"):
            action = "investigate"
            memory["investigation_requested"] = False
        elif score >= threshold and (materially_changed or last_status == "unknown"):
            action = "notify_portfolio_manager"
        elif status == "disputed":
            action = "defer_and_monitor"
        else:
            action = "record_only"
        reactions.append({"event_id": event.event_id, "story_id": story_id, "revision": revision,
                          "action": action, "score": round(score, 5), "evidence": [event.event_id],
                          "status": status})
        last_status = status
    memory.update({"seen_revisions": sorted(seen_revisions), "seen_content": sorted(seen_content),
                   "story_status": last_status,
                   "portfolio_id": portfolio.portfolio_id,
                   "notification_count": int(memory.get("notification_count", 0)) + sum(
                       row["action"] == "notify_portfolio_manager" for row in reactions)})
    return reactions, memory


def process_news_timeline(news_events: tuple[Event, ...], feedback_events: tuple[Event, ...],
                          portfolio: PortfolioExposure) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply news and feedback in event-time order, preserving online state."""
    timeline = sorted(
        [(event, "news") for event in news_events] + [(event, "feedback") for event in feedback_events],
        key=lambda item: (item[0].event_time, item[0].event_id),
    )
    reactions: list[dict[str, Any]] = []
    state: dict[str, Any] = {}
    for event, kind in timeline:
        if kind == "news":
            event_reactions, state = process_news_events((event,), portfolio, state)
            reactions.extend(event_reactions)
        else:
            state = apply_feedback((event,), state)
    return reactions, state


def apply_feedback(events: tuple[Event, ...], state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Update routing state from acknowledgements, corrections, requests, and outcomes."""
    memory = dict(state or {})
    counts = dict(memory.get("feedback_counts", {}))
    threshold = float(memory.get("notification_threshold", 0.15))
    for event in sorted(events, key=lambda item: (item.event_time, item.event_id)):
        kind = event.event_type
        counts[kind] = int(counts.get(kind, 0)) + 1
        if kind == "pm.acknowledgement":
            threshold = max(0.10, threshold - 0.01)
        elif kind == "pm.disagreement":
            threshold = min(0.90, threshold + 0.05)
        elif kind == "pm.request_investigation":
            memory["investigation_requested"] = True
        elif kind == "notification.false_positive":
            threshold = min(0.90, threshold + 0.20)
        elif kind == "portfolio.change" and "rare_earths_exposure" in event.attributes:
            memory["rare_earths_exposure"] = float(event.attributes["rare_earths_exposure"])
        elif kind == "market.realised_outcome":
            memory["last_realised_outcome"] = float(event.attributes["feedback_value"])
    memory.update({"notification_threshold": threshold, "feedback_counts": counts,
                   "feedback_version": int(memory.get("feedback_version", 0)) + len(events)})
    return memory


def synthetic_news_outcomes(seed: int = 11, events: int = 1200) -> tuple[Event, ...]:
    """Chronological synthetic topic and next-session return pairs, for prediction only."""
    rng = np.random.default_rng(seed)
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    result = []
    for index in range(events):
        topic = int(rng.random() < 0.12)
        regime = index >= events * 2 // 3
        effect = 0.012 if not regime else 0.0
        next_return = float(effect * topic + rng.normal(0, 0.02))
        when = start + timedelta(hours=24 * index)
        result.append(Event(
            event_id=stable_id("news-market", str(seed), str(index)),
            event_type="news.topic_observed", event_time=when, source="synthetic-news-market-v1",
            entity_ids=("sector:materials",), attributes={"topic_export_restriction": topic},
            provenance={"generator": "synthetic-news-market-v1", "seed": seed},
        ))
        result.append(Event(
            event_id=stable_id("market-outcome", str(seed), str(index)),
            event_type="market.next_session_return", event_time=when + timedelta(hours=8),
            source="synthetic-market-v1", entity_ids=("index:materials",),
            attributes={"next_session_return": round(next_return, 8), "outcome_for": result[-1].event_id,
                        "market_observation": "next_session"},
            provenance={"generator": "synthetic-news-market-v1", "seed": seed},
        ))
    return tuple(result)


def evaluate_topic_association(events: tuple[Event, ...], *, split_fraction: float = 0.7) -> dict[str, Any]:
    """Estimate a training association and report later predictive usefulness; not causality."""
    topics = [event for event in events if event.event_type == "news.topic_observed"]
    outcomes = {str(event.attributes["outcome_for"]): float(event.attributes["next_session_return"])
                for event in events if event.event_type == "market.next_session_return"}
    joined = [(event, outcomes[event.event_id]) for event in topics if event.event_id in outcomes]
    boundary = int(len(joined) * split_fraction)
    train, test = joined[:boundary], joined[boundary:]
    def difference(rows):
        yes = [y for event, y in rows if event.attributes["topic_export_restriction"] == 1]
        no = [y for event, y in rows if event.attributes["topic_export_restriction"] == 0]
        return (float(np.mean(yes) - np.mean(no)) if yes and no else None)
    learned = difference(train)
    held_out = difference(test)
    predictions = [(learned or 0.0) * int(event.attributes["topic_export_restriction"]) for event, _ in test]
    actual = [actual for _, actual in test]
    return {"train_count": len(train), "test_count": len(test), "training_association": learned,
            "held_out_association": held_out,
            "held_out_mae": float(np.mean(np.abs(np.asarray(predictions) - np.asarray(actual)))) if test else None,
            "evaluation": "chronological; association is predictive evidence, not a causal estimate"}


def synthetic_feedback(seed: int = 11, count: int = 24) -> tuple[Event, ...]:
    rng = np.random.default_rng(seed)
    start = datetime(2025, 2, 1, 8, tzinfo=timezone.utc)
    kinds = ("pm.acknowledgement", "pm.disagreement", "pm.request_investigation",
             "portfolio.change", "notification.false_positive", "market.realised_outcome")
    offsets = (10, 65, 80, 145, 175, 220)
    return tuple(Event(event_id=stable_id("feedback", str(seed), str(i)), event_type=kinds[i % len(kinds)],
                       event_time=start + timedelta(minutes=offsets[i % len(offsets)] +
                                                    360 * (i // len(offsets))),
                       source="synthetic-feedback-v1",
                       entity_ids=("portfolio:demo",), attributes={"feedback_value": float(rng.normal()),
                       "rare_earths_exposure": round(float(rng.uniform(0.0, 0.5)), 4) if kinds[i % len(kinds)] == "portfolio.change" else None,
                       "action_id": f"action-{i // 3}", "seed": seed},
                       provenance={"generator": "synthetic-feedback-v1"}) for i in range(count))


OPS_CAUSES: dict[str, tuple[str, ...]] = {
    "failed_settlement": ("incorrect_instructions", "insufficient_funds", "missing_confirmation",
                          "counterparty_discrepancy", "upstream_failure"),
    "reconciliation_break": ("missing_record", "duplicate_transaction", "timing_difference",
                             "incorrect_mapping", "economic_discrepancy"),
    "collateral_exception": ("unmatched_margin_call", "disputed_valuation", "eligibility_failure",
                             "incorrect_haircut", "missing_movement"),
    "risk_pipeline_failure": ("missing_market_data", "stale_input", "schema_change",
                              "dependency_failure", "model_error", "invalid_output"),
}

OPS_REPAIR_EFFECTS: dict[str, dict[str, Any]] = {
    "incorrect_instructions": {"instruction_match": True},
    "insufficient_funds": {"cash_available": True},
    "missing_confirmation": {"confirmation_received": True},
    "counterparty_discrepancy": {"instruction_match": True},
    "upstream_failure": {"upstream_status": "recovered"},
    "missing_record": {"right_present": True},
    "duplicate_transaction": {"left_count": 1},
    "timing_difference": {"left_time_offset_minutes": 0},
    "incorrect_mapping": {"mapping_valid": True},
    "economic_discrepancy": {"amount_difference": 0.0},
    "unmatched_margin_call": {"call_matched": True},
    "disputed_valuation": {"valuation_disputed": False},
    "eligibility_failure": {"eligible": True},
    "incorrect_haircut": {"haircut_applied": 0.12},
    "missing_movement": {"movement_received": True},
    "missing_market_data": {"required_mark_present": True},
    "stale_input": {"input_age_minutes": 0},
    "schema_change": {"consumer_version": "new"},
    "dependency_failure": {"dependency_status": "available"},
    "model_error": {"execution_status": "succeeded"},
    "invalid_output": {"output_validation": "passed"},
}


@dataclass
class OperationalDataset:
    events: tuple[Event, ...]
    causes_by_event: dict[str, str]


def synthetic_incidents(seed: int = 11, count: int = 40) -> OperationalDataset:
    rng = np.random.default_rng(seed)
    start = datetime(2025, 4, 1, tzinfo=timezone.utc)
    result = []
    causes_by_event = {}
    for index in range(count):
        family = tuple(OPS_CAUSES)[index % len(OPS_CAUSES)]
        causes = OPS_CAUSES[family]
        cause = causes[(index // len(OPS_CAUSES)) % len(causes)]
        recurring = index >= len(causes) * len(OPS_CAUSES)
        symptoms = _incident_symptoms(family, cause, rng)
        event = Event(event_id=stable_id("ops-incident", str(seed), str(index)),
                      event_type=f"ops.{family}", event_time=start + timedelta(minutes=17 * index),
                      source="simulated-operations", entity_ids=(f"tenant-{index % 2}", f"case-{index}"),
                      attributes={"incident_id": f"incident-{index}", "family": family,
                                  "symptoms": symptoms, "severity": int(rng.integers(1, 5)),
                                  "variation": float(rng.normal()), "recurring": recurring,
                                  "requires_human_approval": index % 11 == 0},
                      provenance={"generator": "synthetic-ops-v1", "seed": seed})
        result.append(event)
        causes_by_event[event.event_id] = cause
    return OperationalDataset(tuple(result), causes_by_event)


def _incident_symptoms(family: str, cause: str, rng: np.random.Generator) -> dict[str, Any]:
    symptom_catalog = {
        "incorrect_instructions": {"instruction_match": False, "cash_available": True},
        "insufficient_funds": {"instruction_match": True, "cash_available": False},
        "missing_confirmation": {"instruction_match": True, "confirmation_received": False},
        "counterparty_discrepancy": {"instruction_match": False, "confirmation_received": True},
        "upstream_failure": {"upstream_status": "failed"},
        "missing_record": {"left_present": True, "right_present": False},
        "duplicate_transaction": {"left_count": 2, "right_count": 1},
        "timing_difference": {"left_time_offset_minutes": 7},
        "incorrect_mapping": {"mapping_valid": False},
        "economic_discrepancy": {"amount_difference": 17.25},
        "unmatched_margin_call": {"call_matched": False},
        "disputed_valuation": {"valuation_disputed": True},
        "eligibility_failure": {"eligible": False},
        "incorrect_haircut": {"haircut_applied": 0.08, "haircut_expected": 0.12},
        "missing_movement": {"movement_received": False},
        "missing_market_data": {"required_mark_present": False},
        "stale_input": {"input_age_minutes": 143},
        "schema_change": {"schema_version": "new", "consumer_version": "old"},
        "dependency_failure": {"dependency_status": "unavailable"},
        "model_error": {"execution_status": "exception"},
        "invalid_output": {"output_validation": "failed"},
    }
    known = symptom_catalog.get(cause, {})
    # Add noisy, non-diagnostic context without leaking the hidden cause label.
    return {**known, "queue_depth": int(rng.integers(1, 30)), "source_count": int(rng.integers(1, 4))}


def diagnose_incident(incident: Event) -> str | None:
    """Transparent symptom rule baseline; not an LLM or a learned causal model."""
    symptoms = incident.attributes.get("symptoms", {})
    for cause in OPS_CAUSES[str(incident.attributes["family"])]:
        expected = _symptom_signature(cause)
        if expected and all(symptoms.get(key) == value for key, value in expected.items()):
            return cause
    return None


def _symptom_signature(cause: str) -> dict[str, Any]:
    signatures = {
        "incorrect_instructions": {"instruction_match": False, "cash_available": True},
        "insufficient_funds": {"instruction_match": True, "cash_available": False},
        "missing_confirmation": {"confirmation_received": False},
        "counterparty_discrepancy": {"instruction_match": False, "confirmation_received": True},
        "upstream_failure": {"upstream_status": "failed"},
        "missing_record": {"left_present": True, "right_present": False},
        "duplicate_transaction": {"left_count": 2, "right_count": 1},
        "timing_difference": {"left_time_offset_minutes": 7},
        "incorrect_mapping": {"mapping_valid": False},
        "economic_discrepancy": {"amount_difference": 17.25},
        "unmatched_margin_call": {"call_matched": False},
        "disputed_valuation": {"valuation_disputed": True},
        "eligibility_failure": {"eligible": False},
        "incorrect_haircut": {"haircut_applied": 0.08, "haircut_expected": 0.12},
        "missing_movement": {"movement_received": False},
        "missing_market_data": {"required_mark_present": False},
        "stale_input": {"input_age_minutes": 143},
        "schema_change": {"schema_version": "new", "consumer_version": "old"},
        "dependency_failure": {"dependency_status": "unavailable"},
        "model_error": {"execution_status": "exception"},
        "invalid_output": {"output_validation": "failed"},
    }
    return signatures.get(cause, {})


@dataclass
class SimulatedOperationsSystem:
    """Synthetic operations environment that exposes symptoms, not hidden causes."""
    causes_by_event: dict[str, str]
    symptoms_by_event: dict[str, dict[str, Any]]

    @classmethod
    def create(cls, causes_by_event: dict[str, str]) -> "SimulatedOperationsSystem":
        return cls(dict(causes_by_event), {})

    def execute_repair(self, incident: Event, *, diagnosis: str, proposed_action: str) -> dict[str, Any]:
        incident_id = incident.event_id
        before = dict(self.symptoms_by_event.get(incident_id,
                                                incident.attributes.get("symptoms", {})))
        after = dict(before)
        actual_cause = self.causes_by_event.get(incident_id)
        if diagnosis == actual_cause and proposed_action == f"sandbox:{diagnosis}":
            after.update(OPS_REPAIR_EFFECTS.get(diagnosis, {}))
        self.symptoms_by_event[incident_id] = after
        observed = incident.model_copy(update={
            "attributes": {**incident.attributes, "symptoms": after},
        })
        residual_diagnosis = diagnose_incident(observed)
        return {"symptoms_before": before, "symptoms_after": after,
                "residual_diagnosis": residual_diagnosis}


@dataclass
class RepairSandbox:
    """Closed-loop repair simulator with explicit approval, retry, and rollback states."""
    ledger: list[dict[str, Any]]
    system: SimulatedOperationsSystem

    @classmethod
    def create(cls, system: SimulatedOperationsSystem) -> "RepairSandbox":
        return cls([], system)

    def repair(self, incident: Event, *, diagnosis: str, proposed_action: str,
               external_result: Literal["success", "timeout", "partial", "failure"] = "success",
               approved: bool = False, retries: int = 1) -> dict[str, Any]:
        record = {"incident_id": incident.attributes["incident_id"], "diagnosis": diagnosis,
                  "proposed_action": proposed_action, "attempts": 0, "status": "investigating",
                  "sandbox_only": True, "idempotency_key": incident.event_id, "rolled_back": False}
        if incident.attributes.get("requires_human_approval") and not approved:
            record["status"] = "awaiting_approval"
        else:
            for attempt in range(1, retries + 2):
                record["attempts"] = attempt
                if external_result == "success":
                    outcome = self.system.execute_repair(
                        incident, diagnosis=diagnosis, proposed_action=proposed_action,
                    )
                    record.update(outcome)
                    record["status"] = "verified" if outcome["residual_diagnosis"] is None else "validation_failed"
                    break
                if external_result == "partial":
                    record.update(status="rolled_back", rolled_back=True)
                    break
                if external_result == "failure":
                    record["status"] = "escalated_failure"
                    break
                record["status"] = "retrying"
            if record["status"] == "retrying":
                record["status"] = "escalated_timeout"
        existing_index = next((index for index, item in enumerate(self.ledger)
                               if item["idempotency_key"] == record["idempotency_key"]), None)
        if existing_index is None:
            self.ledger.append(record)
        elif self.ledger[existing_index]["status"] == "awaiting_approval":
            self.ledger[existing_index] = record
        else:
            record = self.ledger[existing_index]
        return dict(record)


@dataclass
class CuratedReaction:
    reaction_id: str
    version: int
    feature: str
    threshold: float
    status: Literal["candidate", "active", "retired", "rejected"]
    validation: dict[str, Any]
    provenance: tuple[str, ...]


def validate_candidate(reaction_id: str, feature: str, train: list[float], held_out: list[float],
                       threshold: float, labels: list[bool], *, version: int = 1) -> CuratedReaction:
    """Gate proposed rules on held-out evidence; score validation is intentionally explicit."""
    if len(held_out) != len(labels) or not held_out:
        raise ValueError("held-out values and labels must be non-empty and aligned")
    predicted = [value >= threshold for value in held_out]
    tp = sum(p and y for p, y in zip(predicted, labels, strict=True))
    fp = sum(p and not y for p, y in zip(predicted, labels, strict=True))
    fn = sum(not p and y for p, y in zip(predicted, labels, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    status = "active" if f1 >= 0.8 and train else "rejected"
    return CuratedReaction(reaction_id, version, feature, threshold, status,
                           {"precision": precision, "recall": recall, "f1": f1,
                            "validation_count": len(held_out)}, tuple())


def state_dependent_observation(event: Event, history: list[Event]) -> dict[str, Any]:
    """Interpret a moderate event against prior exposure movement."""
    previous = [float(item.attributes.get("exposure", 0.0)) for item in history
                if event.entity_ids and item.entity_ids == event.entity_ids]
    current = float(event.attributes.get("exposure", 0.0))
    rises = sum(b > a for a, b in zip(previous, previous[1:]))
    return {"event_id": event.event_id, "stateful": bool(rises >= 2 and previous and current > previous[-1]),
            "prior_count": len(previous), "prior_rises": rises,
            "action": "investigate" if rises >= 2 and previous and current > previous[-1] else "record"}


__all__ = ["PortfolioExposure", "synthetic_news_story", "process_news_events", "synthetic_news_outcomes",
           "process_news_timeline", "evaluate_topic_association", "synthetic_feedback", "apply_feedback",
           "OPS_CAUSES", "OPS_REPAIR_EFFECTS", "OperationalDataset", "synthetic_incidents", "diagnose_incident",
           "SimulatedOperationsSystem",
           "RepairSandbox", "CuratedReaction", "validate_candidate", "state_dependent_observation"]
