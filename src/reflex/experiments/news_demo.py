"""Deterministic synthetic portfolio-news Reaction demonstration."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from reflex.core.models.events import ActionEvent, NativeEvent, ObservationEvent, stable_id
from reflex.core.storage import EventStore, StateStore
from ..contracts.reactions import FunctionRef, ReactionDefinition, ReactionEdge, ReactionNode
from ..contracts.schemas import ExploratoryObservation, SchemaRegistry
from ..runtime import ActionExecutor, FunctionRegistry, ReactionRuntime


def build_portfolio_news_reaction(schemas: SchemaRegistry, state: StateStore,
                                  action_executor: ActionExecutor) -> tuple[ReactionRuntime, ReactionDefinition]:
    """Build the versioned demo graph and register its deterministic functions."""
    _register_demo_schemas(schemas)
    state.set("operational", "portfolio", {
        "manager_id": "pm-synthetic-1",
        "holdings": {"NORTHSTAR": 1200, "ACME": 500, "HARBOR BANK": 800},
        "currency": "USD",
    }, "seed synthetic portfolio")

    functions = FunctionRegistry()

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

    def investigate_relation(event, _state):
        candidates = event.payload.get("candidate_relationships", [])
        records = []
        for index, candidate in enumerate(candidates):
            record = ExploratoryObservation(
                subject=str(candidate.get("subject", "unknown")),
                predicate=str(candidate.get("predicate", "related_to")),
                object=candidate.get("object"),
                evidence=(event.parent_event_ids[0] if event.parent_event_ids else event.event_id,),
                confidence=float(candidate.get("confidence", 0.55)),
                provenance={"skill": "deterministic-news-investigator@1.0", "story_id": event.payload.get("story_id")},
                proposed_classification=str(candidate.get("proposed_classification", "business_relationship")),
            )
            payload = schemas.record_exploratory(record)
            records.append(ObservationEvent(
                event_id=stable_id("exploratory-relation", event.event_id, str(index)),
                event_type="observation.exploratory_relation", event_time=event.event_time,
                entity_ids=(record.subject,), source="skill:deterministic-news-investigator@1.0",
                provenance=dict(record.provenance), parent_event_ids=record.evidence,
                correlation_id=event.correlation_id, causation_id=event.event_id,
                schema_ref=SchemaRegistry.EXPLORATORY_REF, payload=payload,
                lens_id="deterministic-news-investigator", lens_version="1.0", subject=record.subject,
            ))
        return records

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

    def record_hypothesis(event, _snapshot):
        payload = dict(event.payload)
        key = stable_id("knowledge-hypothesis", payload["subject"], payload["predicate"], str(payload["object"]))
        state.set("knowledge", f"hypothesis:{key}", {
            **payload, "status": "hypothesis", "verified": False,
            "source_event_id": event.event_id, "source_function": "record_exploratory_relation@1.0",
        }, "record exploratory relationship as an unverified hypothesis")
        return []

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

    functions.register(FunctionRef(function_id="news-understanding", version="1.0", kind="lens"), understand)
    functions.register(FunctionRef(function_id="portfolio-exposure", version="1.0", kind="lens"), assess_exposure)
    functions.register(FunctionRef(function_id="deterministic-news-investigator", version="1.0", kind="skill"), investigate_relation)
    functions.register(FunctionRef(function_id="portfolio-news-notification", version="1.0", kind="policy"), create_notification)
    functions.register(FunctionRef(function_id="record-exploratory-relation", version="1.0", kind="skill"), record_hypothesis)
    functions.register(FunctionRef(function_id="capture-manager-response", version="1.0", kind="lens"), capture_manager_response)

    definition = ReactionDefinition(
        reaction_id="portfolio_news_intelligence", version="1.0",
        subscriptions=("news.story", "portfolio.manager_response", "observation.exploratory_relation"),
        state_requirements=("operational:portfolio",), permissions=("publish:portfolio_manager",),
        max_steps=50, max_emitted_events=50, max_retries=1,
        nodes=(
            ReactionNode(node_id="understand", function=FunctionRef(function_id="news-understanding", version="1.0", kind="lens"),
                         accepts_event_types=("news.story",), input_schema_refs=("portfolio.news-story@1.0",),
                         output_schema_refs=("reflex.news-understanding@1.0",),
                         edges=(ReactionEdge(target="exposure", event_types=("observation.news_understanding",)),)),
            ReactionNode(node_id="exposure", function=FunctionRef(function_id="portfolio-exposure", version="1.0", kind="lens"),
                         accepts_event_types=("observation.news_understanding",), input_schema_refs=("reflex.news-understanding@1.0",),
                         output_schema_refs=("reflex.portfolio-exposure@1.0",), state_requirements=("operational:portfolio",),
                         edges=(ReactionEdge(target="investigate", event_types=("observation.portfolio_exposure",)),
                                ReactionEdge(target="notify", event_types=("observation.portfolio_exposure",)))),
            ReactionNode(node_id="investigate", function=FunctionRef(function_id="deterministic-news-investigator", version="1.0", kind="skill"),
                         accepts_event_types=("observation.portfolio_exposure",),
                         output_schema_refs=(SchemaRegistry.EXPLORATORY_REF,)),
            ReactionNode(node_id="notify", function=FunctionRef(function_id="portfolio-news-notification", version="1.0", kind="policy"),
                         accepts_event_types=("observation.portfolio_exposure",)),
            ReactionNode(node_id="record-hypothesis", function=FunctionRef(function_id="record-exploratory-relation", version="1.0", kind="skill"),
                         accepts_event_types=("observation.exploratory_relation",), input_schema_refs=(SchemaRegistry.EXPLORATORY_REF,)),
            ReactionNode(node_id="manager-response", function=FunctionRef(function_id="capture-manager-response", version="1.0", kind="lens"),
                         accepts_event_types=("portfolio.manager_response",), input_schema_refs=("portfolio.manager-response@1.0",),
                         output_schema_refs=("reflex.manager-response@1.0",)),
        ),
    )
    runtime = ReactionRuntime(action_executor.event_store, state, functions, schemas, action_executor)
    return runtime, definition


def run_news_demo(out_dir: str | Path) -> dict[str, Any]:
    """Run relevant, irrelevant, duplicate, revision, retry, replay and skill cases."""
    started = perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    event_store = EventStore(out / "events.duckdb")
    state = StateStore(out / "state.sqlite")
    published: list[dict[str, Any]] = []
    fail_once = {"news-100": True}

    def publish(arguments: dict[str, Any]) -> dict[str, Any]:
        if arguments["story_id"] in fail_once and fail_once[arguments["story_id"]]:
            fail_once[arguments["story_id"]] = False
            raise RuntimeError("synthetic transient publisher outage")
        message = {**arguments, "message_id": f"msg-{len(published) + 1}"}
        published.append(message)
        return {"message_id": message["message_id"], "delivered_to": arguments["manager_id"]}

    schemas = SchemaRegistry(state)
    action_executor = ActionExecutor(event_store, state, schemas)
    action_executor.register("publish_to_pm", "1.0", "publish:portfolio_manager", publish,
                             argument_schema_ref="reflex.portfolio-notification@1.0",
                             result_schema_ref="reflex.portfolio-notification-result@1.0")
    runtime, definition = build_portfolio_news_reaction(schemas, state, action_executor)

    inputs, response = benchmark_inputs()
    runs = {event.event_id: runtime.process(event, definition) for event in inputs}
    response_run = runtime.process(response, definition)

    side_effect_count_before_replay = len(published)
    replay_state = StateStore(out / "replay-state.sqlite")
    replay_state.set("operational", "portfolio", state.get("operational", "portfolio"), "pin replay state snapshot")
    replay_schemas = SchemaRegistry(replay_state)
    replay_actions = ActionExecutor(event_store, replay_state, replay_schemas)
    replay_actions.register("publish_to_pm", "1.0", "publish:portfolio_manager", publish,
                            argument_schema_ref="reflex.portfolio-notification@1.0",
                            result_schema_ref="reflex.portfolio-notification-result@1.0")
    replay_runtime, replay_definition = build_portfolio_news_reaction(replay_schemas, replay_state, replay_actions)
    replay_runs = [replay_runtime.process(event, replay_definition, replay=True,
                                          state_snapshot=runs[event.event_id].state_snapshot) for event in inputs]
    replay_side_effect_count = len(published) - side_effect_count_before_replay

    stored = event_store.events()
    action_events = [event for event in stored if isinstance(event, ActionEvent)]
    outcomes = [event for event in stored if event.event_type in {"action.failed", "action.succeeded"}]
    successful_actions = {event.action_event_id for event in stored
                          if getattr(event, "event_type", None) == "action.succeeded"}
    hypotheses = {key: value for key, value in state.snapshot("knowledge").items() if key.startswith("hypothesis:")}
    type_by_id = {event.event_id: event.event_type for event in stored}
    live_action_ids = [event_id for run in runs.values() for event_id in run.emitted_event_ids
                       if type_by_id.get(event_id) == "action.notify_portfolio_manager"]
    replay_action_ids = [event_id for run in replay_runs for event_id in run.emitted_event_ids
                         if type_by_id.get(event_id) == "action.notify_portfolio_manager"]
    duplicate_run = runs["news-100-duplicate"]
    result = {
        "reaction": f"{definition.reaction_id}@{definition.version}",
        "synthetic_only": True,
        "scenarios": {
            "relevant_news_alerted": any(item.event_type == "action.notify_portfolio_manager" and
                                          item.arguments.get("story_id") == "news-100" and item.arguments.get("revision") == 0 for item in action_events),
            "irrelevant_news_no_alert": not any(item.arguments.get("story_id") == "news-200" for item in action_events),
            "duplicate_suppressed": not any(type_by_id.get(event_id) == "action.notify_portfolio_manager"
                                             for event_id in duplicate_run.emitted_event_ids),
            "updated_news_revised_observation": any(event.event_type == "observation.portfolio_exposure" and
                                                   event.payload.get("story_id") == "news-100" and event.payload.get("revision") == 1 for event in stored),
            "failed_action_retried": {"failed_attempts": sum(event.event_type == "action.failed" for event in outcomes),
                                      "successful_attempts": sum(event.event_type == "action.succeeded" for event in outcomes)},
            "manager_response_observed": response_run.status == "succeeded" and "observation.manager_response" in
                                         [event.event_type for event in stored],
            "unknown_relationship_recorded_as_hypothesis": any(value.get("verified") is False for value in hypotheses.values()),
            "replay_external_side_effects": replay_side_effect_count,
            "replay_decisions_match": sorted(set(live_action_ids)) == sorted(set(replay_action_ids)),
        },
        "action_event_count": len(action_events), "publisher_deliveries": len(successful_actions),
        "action_events": [{"event_id": event.event_id, "story_id": event.arguments.get("story_id"),
                           "revision": event.arguments.get("revision"), "parent_event_ids": list(event.parent_event_ids)}
                          for event in action_events],
        "action_results": [{"event_id": event.event_id, "action_event_id": event.action_event_id,
                            "status": event.status, "attempt": event.attempt}
                           for event in stored if hasattr(event, "action_event_id")],
        "live_action_event_ids": live_action_ids, "replay_action_event_ids": replay_action_ids,
        "input_hash": benchmark_input_hash(), "wall_seconds": perf_counter() - started,
        "live_runs": {key: value.model_dump(mode="json") for key, value in runs.items()},
        "replay_runs": [value.model_dump(mode="json") for value in replay_runs],
        "event_count": event_store.count_events(),
        "database": str(out / "events.duckdb"),
    }
    (out / "portfolio-news-demo.json").write_text(__import__("json").dumps(result, indent=2))
    replay_state.close()
    state.close()
    event_store.close()
    return result


def benchmark_inputs() -> tuple[list[NativeEvent], NativeEvent]:
    """Shared, fixed information bundle used by the REFLEX demo and A/B/C harness."""
    start = datetime(2026, 1, 12, 9, tzinfo=timezone.utc)
    inputs = [
        _story("news-100-initial", "news-100", 0, start, "Northstar raises its earnings outlook",
               ["NORTHSTAR"], ["earnings", "guidance"]),
        _story("news-200-irrelevant", "news-200", 0, start + timedelta(minutes=4),
               "Redwood Mining opens a new site", ["REDWOOD MINING"], ["operations"]),
        _story("news-100-duplicate", "news-100", 0, start + timedelta(minutes=5),
               "Northstar raises its earnings outlook", ["NORTHSTAR"], ["earnings", "guidance"]),
        _story("news-100-revision-1", "news-100", 1, start + timedelta(minutes=15),
               "Northstar raises its outlook again after record demand", ["NORTHSTAR"], ["earnings", "demand"]),
        _story("news-300-unknown-relation", "news-300", 0, start + timedelta(minutes=20),
               "Nebula Tech may supply components to Northstar", ["NEBULA TECH"], ["supply chain"],
               candidate_relationships=[{"subject": "NEBULA TECH", "predicate": "may_supply_to",
                                         "object": "NORTHSTAR", "confidence": 0.61,
                                         "proposed_classification": "supplier_relationship"}]),
    ]
    response = NativeEvent(
        event_id="manager-response-1", event_type="portfolio.manager_response", event_time=start + timedelta(minutes=30),
        entity_ids=("pm-synthetic-1",), source="synthetic:portfolio-manager",
        correlation_id="story/news-100", causation_id="news-100-revision-1",
        schema_ref="portfolio.manager-response@1.0",
        payload={"manager_id": "pm-synthetic-1", "story_id": "news-100", "response": "Please monitor supplier exposure.", "rating": 1},
    )
    return inputs, response


def benchmark_input_hash() -> str:
    inputs, response = benchmark_inputs()
    bundle = {"portfolio": {"manager_id": "pm-synthetic-1", "holdings": {"NORTHSTAR": 1200, "ACME": 500, "HARBOR BANK": 800}},
              "events": [event.model_dump(mode="json") for event in (*inputs, response)]}
    encoded = json.dumps(bundle, sort_keys=True, separators=(",", ":"))
    return sha256(encoded.encode("utf-8")).hexdigest()


def _story(event_id: str, story_id: str, revision: int, at: datetime, headline: str,
           companies: list[str], themes: list[str], *, candidate_relationships: list[dict[str, Any]] | None = None) -> NativeEvent:
    payload = {"story_id": story_id, "revision": revision, "headline": headline, "companies": companies,
               "themes": themes, "relationships": [], "candidate_relationships": candidate_relationships or []}
    return NativeEvent(event_id=event_id, event_type="news.story", event_time=at,
                       entity_ids=tuple(companies), source="synthetic:portfolio-news",
                       correlation_id=f"story/{story_id}", idempotency_key=f"{story_id}:{revision}:{event_id}",
                       schema_ref="portfolio.news-story@1.0", payload=payload, attributes=payload,
                       provenance={"synthetic": True, "label": "not real market news"})


def _register_demo_schemas(schemas: SchemaRegistry) -> None:
    string_array = {"type": "array", "items": {"type": "string"}}
    schemas.register("portfolio.news-story", "1.0", {
        "type": "object", "required": ["story_id", "revision", "headline", "companies", "themes"],
        "properties": {"story_id": {"type": "string"}, "revision": {"type": "integer"},
                       "headline": {"type": "string"}, "companies": string_array, "themes": string_array,
                       "relationships": {"type": "array"}, "candidate_relationships": {"type": "array"}},
    })
    schemas.register("reflex.news-understanding", "1.0", {
        "type": "object", "required": ["story_id", "revision", "headline", "companies", "themes"],
        "properties": {"story_id": {"type": "string"}, "revision": {"type": "integer"},
                       "headline": {"type": "string"}, "companies": string_array, "themes": string_array,
                       "relationships": {"type": "array"}, "candidate_relationships": {"type": "array"}},
    })
    schemas.register("reflex.portfolio-exposure", "1.0", {
        "type": "object", "required": ["story_id", "revision", "relevant", "held_companies"],
        "properties": {"story_id": {"type": "string"}, "revision": {"type": "integer"},
                       "headline": {"type": "string"}, "companies": string_array,
                       "held_companies": string_array, "exposure_value": {"type": "number"},
                       "relevant": {"type": "boolean"}, "candidate_relationships": {"type": "array"}},
    })
    schemas.register("portfolio.manager-response", "1.0", {
        "type": "object", "required": ["manager_id", "story_id", "response"],
        "properties": {"manager_id": {"type": "string"}, "story_id": {"type": "string"},
                       "response": {"type": "string"}, "rating": {"type": "integer"}},
    })
    schemas.register("reflex.portfolio-notification", "1.0", {
        "type": "object", "required": ["story_id", "revision", "manager_id", "headline", "held_companies"],
        "properties": {"story_id": {"type": "string"}, "revision": {"type": "integer"},
                       "manager_id": {"type": "string"}, "headline": {"type": "string"},
                       "held_companies": string_array},
    })
    schemas.register("reflex.portfolio-notification-result", "1.0", {
        "type": "object", "required": ["message_id", "delivered_to"],
        "properties": {"message_id": {"type": "string"}, "delivered_to": {"type": "string"}},
    })
    schemas.register("reflex.manager-response", "1.0", {
        "type": "object", "required": ["story_id", "response"],
        "properties": {"story_id": {"type": "string"}, "response": {"type": "string"}, "rating": {"type": "integer"}},
    })


__all__ = ["build_portfolio_news_reaction", "run_news_demo"]
