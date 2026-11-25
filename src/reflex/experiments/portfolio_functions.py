"""Stateless Reaction lenses and notification policy for the portfolio-news example."""

from reflex.contracts.reactions import FunctionRef
from reflex.core.models.events import ActionEvent, ObservationEvent, stable_id
from reflex.registry import FunctionSpec


def understand(event, _state):
    body = dict(event.payload)
    payload = {
        "story_id": body["story_id"], "revision": body.get("revision", 0),
        "headline": body["headline"], "companies": body.get("companies", []),
        "themes": body.get("themes", []), "relationships": body.get("relationships", []),
        "candidate_relationships": body.get("candidate_relationships", []),
    }
    return [ObservationEvent(
        event_id=stable_id("news-understanding", event.event_id), event_type="observation.news_understanding",
        event_time=event.event_time, entity_ids=event.entity_ids, source="lens:news-understanding@1.0",
        provenance={"input_story_id": body["story_id"], "revision": body.get("revision", 0)},
        parent_event_ids=(event.event_id,), correlation_id=event.correlation_id or f"story/{body['story_id']}",
        causation_id=event.event_id, schema_ref="reflex.news-understanding@1.0", payload=payload,
        lens_id="news-understanding", lens_version="1.0", subject=body["story_id"],
    )]


def assess_exposure(event, snapshot):
    body = dict(event.payload)
    portfolio = snapshot.get("portfolio", {})
    holdings = portfolio.get("holdings", {})
    entities = body.get("companies", [])
    directly_held = sorted(set(entities) & set(holdings))
    payload = {
        "story_id": body["story_id"], "revision": body.get("revision", 0),
        "headline": body["headline"], "companies": entities,
        "held_companies": directly_held,
        "exposure_value": sum(float(holdings[item]) for item in directly_held),
        "relevant": bool(directly_held),
        "candidate_relationships": body.get("candidate_relationships", []),
    }
    return [ObservationEvent(
        event_id=stable_id("portfolio-exposure", event.event_id), event_type="observation.portfolio_exposure",
        event_time=event.event_time, entity_ids=tuple(directly_held), source="lens:portfolio-exposure@1.0",
        provenance={"portfolio_id": portfolio.get("manager_id"), "holdings_source": "synthetic_state"},
        parent_event_ids=(event.event_id,), correlation_id=event.correlation_id,
        causation_id=event.event_id, schema_ref="reflex.portfolio-exposure@1.0", payload=payload,
        lens_id="portfolio-exposure", lens_version="1.0", subject=portfolio.get("manager_id", "unknown"),
    )]


def create_notification(event, snapshot):
    body = dict(event.payload)
    if not body.get("relevant"):
        return []
    revision = int(body.get("revision", 0))
    action_id = stable_id("portfolio-news-notification", body["story_id"], str(revision))
    prior = snapshot.get(f"action:{action_id}", {})
    if prior:
        return []
    portfolio = snapshot.get("portfolio", {})
    return [ActionEvent(
        event_id=action_id, event_type="action.notify_portfolio_manager", event_time=event.event_time,
        entity_ids=event.entity_ids, source="policy:portfolio-news-notification@1.0",
        provenance={"policy": "portfolio-news-notification@1.0", "story_id": body["story_id"]},
        parent_event_ids=(event.event_id,), correlation_id=event.correlation_id,
        causation_id=event.event_id, idempotency_key=f"{body['story_id']}:{revision}",
        access_scope=("portfolio-manager",), action_function="publish_to_pm@1.0",
        required_permission="publish:portfolio_manager", retry_limit=1,
        arguments={"story_id": body["story_id"], "revision": revision,
                   "manager_id": portfolio.get("manager_id"), "headline": body["headline"],
                   "held_companies": body["held_companies"]},
    )]


def capture_manager_response(event, _state):
    return [ObservationEvent(
        event_id=stable_id("manager-response", event.event_id),
        event_type="observation.manager_response", event_time=event.event_time,
        entity_ids=event.entity_ids, source="lens:manager-response@1.0",
        provenance={"response_to": event.payload.get("story_id")},
        parent_event_ids=(event.event_id,), correlation_id=event.correlation_id,
        causation_id=event.event_id, schema_ref="reflex.manager-response@1.0",
        payload={"story_id": event.payload.get("story_id"), "response": event.payload.get("response"),
                 "rating": event.payload.get("rating")},
        lens_id="manager-response", lens_version="1.0", subject=event.payload.get("manager_id", "unknown"),
    )]


def portfolio_functions() -> tuple[FunctionSpec, ...]:
    """Provide these versions to a running REFLEX host through package metadata."""
    return (
        FunctionSpec(FunctionRef(function_id="news-understanding", version="1.0", kind="lens"), understand),
        FunctionSpec(FunctionRef(function_id="portfolio-exposure", version="1.0", kind="lens"), assess_exposure),
        FunctionSpec(FunctionRef(function_id="portfolio-news-notification", version="1.0", kind="policy"), create_notification),
        FunctionSpec(FunctionRef(function_id="capture-manager-response", version="1.0", kind="lens"), capture_manager_response),
    )
