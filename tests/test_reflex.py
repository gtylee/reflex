from datetime import datetime, timezone
import json

import duckdb
import pytest

from reflex.core.models import ActionEvent, ActionResultEvent, NativeEvent, Observation, ObservationEvent, stable_id
from reflex.core.storage import EventStore, StateStore
from reflex.benchmark import benchmark_manifest, report, run_reflex_variant
from reflex.contracts import (
    ExploratoryObservation,
    FunctionRef,
    ReactionDefinition,
    ReactionEdge,
    ReactionNode,
    SchemaRegistry,
)
from reflex.news_demo import run_news_demo
from reflex.ingestion import GenericNewsAdapter
from reflex.runtime import ActionExecutor, FunctionRegistry, ReactionRuntime


AT = datetime(2026, 1, 1, tzinfo=timezone.utc)


def test_typed_events_share_one_immutable_event_ledger_and_legacy_observations_adapt(tmp_path):
    store = EventStore(tmp_path / "events.duckdb")
    native = NativeEvent(event_id="native", event_type="news.story", event_time=AT,
                         payload={"headline": "Synthetic story"}, correlation_id="story/1")
    observation = ObservationEvent(event_id="observation", event_type="observation.news", event_time=AT,
                                  lens_id="news-lens", lens_version="1.0", subject="story/1",
                                  payload={"themes": ["earnings"]}, parent_event_ids=("native",))
    action = ActionEvent(event_id="action", event_type="action.notify", event_time=AT,
                         action_function="notify@1.0", required_permission="notify", arguments={"message": "alert"},
                         parent_event_ids=("observation",))
    outcome = ActionResultEvent(event_id="outcome", event_type="action.succeeded", event_time=AT,
                                action_event_id="action", status="succeeded", attempt=1)
    assert store.add_events([native, observation, action, outcome]) == 4
    loaded = {event.event_id: event for event in store.events()}
    assert isinstance(loaded["native"], NativeEvent)
    assert isinstance(loaded["observation"], ObservationEvent)
    assert isinstance(loaded["action"], ActionEvent)
    assert isinstance(loaded["outcome"], ActionResultEvent)
    assert loaded["observation"].parent_event_ids == ("native",)
    assert store.add_events(loaded.values()) == 0

    legacy = Observation(observation_id="legacy-observation", observation_type="score", lens_id="old-lens",
                         lens_version="1", subject="acct-1", window_start=AT, window_end=AT,
                         value={"score": 0.8}, supporting_event_ids=("native",))
    assert store.save_observations([legacy]) == 1
    assert store.load_observations()[0].observation_id == legacy.observation_id
    adapted = {event.event_id: event for event in store.events()}[legacy.observation_id]
    assert isinstance(adapted, ObservationEvent)
    assert adapted.payload["value"] == {"score": 0.8}
    assert [event.event_id for event in store.native_events()] == ["native"]
    assert store.count_native_events() == 1
    store.close()


def test_existing_reflex_database_opens_with_native_event_defaults(tmp_path):
    path = tmp_path / "old-events.duckdb"
    connection = duckdb.connect(str(path))
    connection.execute("""CREATE TABLE events (
        event_id VARCHAR PRIMARY KEY, event_type VARCHAR NOT NULL, event_time TIMESTAMPTZ NOT NULL,
        entity_ids JSON NOT NULL, attributes JSON NOT NULL, source VARCHAR NOT NULL,
        provenance JSON NOT NULL, relationships JSON NOT NULL)""")
    connection.execute("INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                       ["old", "payment", AT, '["acct-1"]', '{"amount": 10}', "legacy",
                        "{}", "[]"])
    connection.close()
    store = EventStore(path)
    event = store.events()[0]
    assert event.event_id == "old" and event.event_kind == "native"
    assert event.payload == {}
    assert store.count_native_events() == 1
    store.close()


def test_schema_registry_validates_exploration_and_promotion_is_explicit(tmp_path):
    state = StateStore(tmp_path / "state.sqlite")
    registry = SchemaRegistry(state)
    record = ExploratoryObservation(subject="NEBULA", predicate="may_supply_to", object="NORTHSTAR",
                                    evidence=("event-1",), confidence=0.61,
                                    provenance={"source": "synthetic-news"},
                                    proposed_classification="supplier_relationship")
    payload = registry.record_exploratory(record)
    registry.validate(SchemaRegistry.EXPLORATORY_REF, payload)
    with pytest.raises(ValueError, match="explicit approval"):
        registry.register("reflex.exploratory.supplier", "1.0", {"type": "object", "properties": {}})
    promoted = registry.promote_exploratory("domain.supplier-relationship", "1.0",
        {"type": "object", "required": ["supplier"], "properties": {"supplier": {"type": "string"}}},
        approved_by="portfolio-reviewer")
    registry.validate(promoted, {"supplier": "NEBULA"})
    assert state.get("knowledge", f"schema-promotion:{promoted}")["policy_authorized"] is False
    state.close()


def test_generic_news_adapter_produces_correlated_native_events_for_future_sources():
    adapter = GenericNewsAdapter("historical-news")
    record = {"story_id": "story-77", "source_record_id": "row-77", "revision": 1,
              "headline": "A synthetic company updates guidance", "published_at": "2026-01-01T09:30:00Z",
              "companies": ["NORTHSTAR"], "themes": ["earnings"], "source_url": "https://example.invalid/story-77"}
    first = adapter.to_event(record)
    second = adapter.to_event(record)
    assert first == second
    assert first.event_type == "news.story" and first.event_kind == "native"
    assert first.correlation_id == "news/historical-news/story-77"
    assert first.provenance["source_record_id"] == "row-77"
    assert first.schema_ref == "portfolio.news-story@1.0"


def test_action_executor_records_fail_retry_idempotency_and_replay_skip(tmp_path):
    store = EventStore(tmp_path / "events.duckdb")
    state = StateStore(tmp_path / "state.sqlite")
    attempts = []

    def publish(arguments):
        attempts.append(arguments["message"])
        if len(attempts) == 1:
            raise RuntimeError("temporary outage")
        return {"message_id": "message-1"}

    executor = ActionExecutor(store, state)
    executor.register("publish", "1.0", "publish", publish)
    event = ActionEvent(event_id="notify-1", event_type="action.notify", event_time=AT,
                        action_function="publish@1.0", required_permission="publish", retry_limit=1,
                        idempotency_key="notice-1", arguments={"message": "hello"})
    result = executor.execute(event, permissions={"publish"})
    assert result is not None and result.status == "succeeded" and result.attempt == 2
    assert attempts == ["hello", "hello"]
    outcomes = [item for item in store.events() if isinstance(item, ActionResultEvent)]
    assert [(item.status, item.attempt) for item in outcomes] == [("failed", 1), ("succeeded", 2)]
    assert executor.execute(event, permissions={"publish"}).event_id == result.event_id
    assert attempts == ["hello", "hello"]

    replay_state = StateStore(tmp_path / "replay.sqlite")
    replay_executor = ActionExecutor(store, replay_state)
    replay_executor.register("publish", "1.0", "publish", publish)
    assert replay_executor.execute(event, permissions={"publish"}, replay=True) is None
    assert replay_state.get("execution", "action:notify-1")["status"] == "skipped_replay"
    assert attempts == ["hello", "hello"]
    denied = event.model_copy(update={"event_id": "notify-denied"})
    with pytest.raises(PermissionError):
        executor.execute(denied, permissions=set())
    assert any(item.event_id == "notify-denied" for item in store.events())
    replay_state.close()
    state.close()
    store.close()


def test_reaction_graph_fanout_reentry_cycle_budget_and_run_idempotency(tmp_path):
    store = EventStore(tmp_path / "events.duckdb")
    state = StateStore(tmp_path / "state.sqlite")
    registry = FunctionRegistry()
    calls = {"root": 0, "left": 0, "right": 0, "audit": 0}

    def output(event_type, suffix, payload=None):
        return NativeEvent(event_id=stable_id(suffix, payload.get("n", "") if payload else ""),
                           event_type=event_type, event_time=AT, payload=payload or {})

    def root(event, _snapshot):
        calls["root"] += 1
        return [output("branch.left", "left-result"), output("branch.right", "right-result")]

    def left(_event, _snapshot):
        calls["left"] += 1
        return [output("observation.complete", "complete-result")]

    def right(_event, _snapshot):
        calls["right"] += 1
        return []

    def audit(_event, _snapshot):
        calls["audit"] += 1
        return []

    refs = {name: FunctionRef(function_id=name, version="1.0", kind="lens")
            for name in ("root", "left", "right", "audit")}
    for name, function in (("root", root), ("left", left), ("right", right), ("audit", audit)):
        registry.register(refs[name], function)
    definition = ReactionDefinition(
        reaction_id="fanout", version="1.0", subscriptions=("source", "observation.complete"),
        nodes=(
            ReactionNode(node_id="root", function=refs["root"], accepts_event_types=("source",),
                         edges=(ReactionEdge(target="left", event_types=("branch.left",)),
                                ReactionEdge(target="right", event_types=("branch.right",)))),
            ReactionNode(node_id="left", function=refs["left"], accepts_event_types=("branch.left",)),
            ReactionNode(node_id="right", function=refs["right"], accepts_event_types=("branch.right",)),
            ReactionNode(node_id="audit", function=refs["audit"], accepts_event_types=("observation.complete",)),
        ), max_steps=10)
    runtime = ReactionRuntime(store, state, registry)
    source = NativeEvent(event_id="source-1", event_type="source", event_time=AT)
    result = runtime.process(source, definition)
    assert result.status == "succeeded"
    assert runtime.load_definition("fanout", "1.0") == definition
    assert calls == {"root": 1, "left": 1, "right": 1, "audit": 1}
    assert result.state_snapshot_hash
    assert runtime.process(source, definition) == result
    assert calls == {"root": 1, "left": 1, "right": 1, "audit": 1}
    protected = NativeEvent(event_id="source-private", event_type="source", event_time=AT,
                            access_scope=("restricted",))
    rejected = runtime.process(protected, definition)
    assert rejected.status == "skipped"
    assert rejected.skip_reason == "Reaction is not granted the event access scope"
    assert calls == {"root": 1, "left": 1, "right": 1, "audit": 1}

    loop_registry = FunctionRegistry()
    loop_ref = FunctionRef(function_id="step", version="1.0", kind="lens")

    def step(event, _snapshot):
        count = event.payload.get("n", 0) + 1
        return [NativeEvent(event_id=stable_id("loop", str(count)), event_type="cycle", event_time=AT,
                            payload={"n": count})]

    loop_registry.register(loop_ref, step)
    loop = ReactionDefinition(reaction_id="loop", version="1.0", subscriptions=("cycle",), entry_nodes=("step",), max_steps=3,
        nodes=(ReactionNode(node_id="step", function=loop_ref, accepts_event_types=("cycle",),
                            edges=(ReactionEdge(target="step", event_types=("cycle",)),)),))
    loop_result = ReactionRuntime(store, state, loop_registry).process(
        NativeEvent(event_id="cycle-0", event_type="cycle", event_time=AT), loop)
    assert loop_result.status == "budget_exceeded"
    assert loop_result.nodes_executed == 3
    store.close()
    state.close()


def test_portfolio_news_end_to_end_synthetic_scenarios(tmp_path):
    result = run_news_demo(tmp_path / "demo")
    scenarios = result["scenarios"]
    assert scenarios["relevant_news_alerted"]
    assert scenarios["irrelevant_news_no_alert"]
    assert scenarios["duplicate_suppressed"]
    assert scenarios["updated_news_revised_observation"]
    assert scenarios["failed_action_retried"] == {"failed_attempts": 1, "successful_attempts": 2}
    assert scenarios["manager_response_observed"]
    assert scenarios["unknown_relationship_recorded_as_hypothesis"]
    assert scenarios["replay_external_side_effects"] == 0
    assert scenarios["replay_decisions_match"]
    assert len(result["action_events"]) == 2
    assert result["publisher_deliveries"] == 2


def test_abc_benchmark_records_only_actual_reflex_run(tmp_path):
    protocol_path = tmp_path / "abc.json"
    protocol_path.write_text(json.dumps(benchmark_manifest(seeds=[11, 23])))
    run = run_reflex_variant(protocol_path, tmp_path / "c-run", seed=11)
    summary = report(protocol_path)
    assert run["variant"] == "C"
    assert run["metrics"]["alert_precision"] == 1.0
    assert summary["variants"]["C"]["runs"] == 1
    assert summary["baseline_status"]["A"] == "not recorded"
    assert summary["baseline_status"]["B"] == "not recorded"
    assert summary["baseline_status"]["C"] == "partially recorded"
    assert summary["variants"]["C"]["missing_seeds"] == [23]
    assert summary["comparisons_ready"] is False
