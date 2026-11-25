"""Exercise live provider discovery through installed package metadata and the Reaction runtime."""

from datetime import UTC, datetime

import pytest

from reflex.contracts.reactions import FunctionRef, ReactionDefinition, ReactionEdge, ReactionNode
from reflex.core.models.events import NativeEvent
from reflex.core.storage import EventStore, StateStore
from reflex.registry import FunctionRegistry, FunctionSpec
from reflex.runtime import ReactionRuntime


def test_load_external_library_after_runtime_start(tmp_path, monkeypatch):
    store = EventStore(tmp_path / "events.duckdb")
    state = StateStore(tmp_path / "state.sqlite")
    try:
        registry = FunctionRegistry()
        runtime = ReactionRuntime(store, state, registry)
        lens = FunctionRef(function_id="external-lens", version="1", kind="lens")
        policy = FunctionRef(function_id="external-policy", version="1", kind="policy")
        condition = FunctionRef(function_id="allow", version="1", kind="condition")
        graph = ReactionDefinition(
            reaction_id="external", version="1", subscriptions=("input",),
            nodes=(
                ReactionNode(node_id="detect", function=lens,
                             edges=(ReactionEdge(target="review", condition_ref=condition.ref),)),
                ReactionNode(node_id="review", function=policy),
            ),
        )
        trigger = NativeEvent(event_type="input", event_time=datetime.now(UTC))
        # Even a missing downstream policy must be caught before the lens emits an event.
        def detect(event, _state):
            return [NativeEvent(event_type="finding", event_time=event.event_time)]

        registry.register(lens, detect)
        registry.register(condition, lambda event, state: True)
        with pytest.raises(KeyError, match="external-policy"):
            runtime.process(trigger, graph)
        assert [event.event_id for event in store.events()] == [trigger.event_id]

        package = tmp_path / "test_review_plugin.py"
        package.write_text(
            "from reflex.registry import FunctionSpec\n"
            "from reflex.contracts.reactions import FunctionRef\n"
            "def review(event, snapshot): return []\n"
            "def provide():\n"
            "    ref = FunctionRef(function_id='external-policy', version='1', kind='policy')\n"
            "    return [FunctionSpec(ref, review)]\n"
        )
        dist = tmp_path / "test_review_plugin-1.dist-info"
        dist.mkdir()
        (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: test-review-plugin\nVersion: 1\n")
        (dist / "entry_points.txt").write_text(
            "[reflex_intelligence.functions]\nexternal-review = test_review_plugin:provide\n"
        )
        monkeypatch.syspath_prepend(str(tmp_path))
        assert "external-review" in registry.available_libraries()
        assert registry.load_library("external-review") == (policy,)
        assert registry.load_library("external-review") == (policy,)
        run = runtime.process(trigger, graph)
        assert run.status == "succeeded"
        assert run.nodes_executed == 2
        assert set(run.function_versions) == {lens.ref, policy.ref, condition.ref}
    finally:
        state.close()
        store.close()


def test_conflicting_batch_keeps_original_registry():
    registry = FunctionRegistry()
    old = FunctionRef(function_id="threshold", version="1", kind="lens")
    new = old.model_copy(update={"version": "2"})

    def original(event, state):
        return []

    registry.register(old, original)
    with pytest.raises(ValueError, match="already registered"):
        registry.register_many((FunctionSpec(new, original), FunctionSpec(old, lambda e, s: [])))
    with pytest.raises(KeyError):
        registry.resolve(new)
    assert registry.resolve(old).function is original
    registry.register(new, original)
    assert registry.resolve(new).ref == new
