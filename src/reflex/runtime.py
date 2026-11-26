"""Bounded event-driven Reaction runtime and allowlisted action executor."""

from __future__ import annotations

from copy import deepcopy
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from hashlib import sha256
import json
from typing import Any, Callable, Iterable, Literal

from pydantic import Field

from reflex.core.models.base import FrozenModel
from reflex.core.models.events import ActionEvent, ActionResultEvent, Event, stable_id
from reflex.core.storage import EventStore, StateStore
from .contracts.reactions import FunctionRef, ReactionDefinition, ReactionEdge, ReactionNode
from .contracts.schemas import SchemaRegistry
from .registry import FunctionRegistry, FunctionSpec, ReactionFunction


ActionFunction = Callable[..., dict[str, Any] | None]


@dataclass(frozen=True)
class ActionFunctionSpec:
    function_id: str
    version: str
    permission: str
    function: ActionFunction
    argument_schema_ref: str | None = None
    result_schema_ref: str | None = None
    pass_idempotency_key: bool = False
    provider: str | None = None
    provider_version: str | None = None

    @property
    def ref(self) -> str:
        return f"{self.function_id}@{self.version}"


class ActionExecutor:
    """Executes registered action functions and writes immutable outcome events."""

    def __init__(self, event_store: EventStore, state: StateStore, schemas: SchemaRegistry | None = None):
        self.event_store = event_store
        self.state = state
        self.schemas = schemas
        self._actions: dict[str, ActionFunctionSpec] = {}

    def register(self, function_id: str, version: str, permission: str, function: ActionFunction, *,
                 argument_schema_ref: str | None = None, result_schema_ref: str | None = None,
                 pass_idempotency_key: bool = False,
                 provider: str | None = None, provider_version: str | None = None) -> None:
        spec = ActionFunctionSpec(function_id, version, permission, function, argument_schema_ref,
                                  result_schema_ref, pass_idempotency_key, provider, provider_version)
        if spec.ref in self._actions:
            raise ValueError(f"action function {spec.ref!r} is already registered")
        self._actions[spec.ref] = spec

    def provider_details(self, function_ref: str) -> dict[str, str | None] | None:
        """Return package metadata for an action function registered by the host."""
        spec = self._actions.get(function_ref)
        if spec is None:
            return None
        return {"provider": spec.provider, "provider_version": spec.provider_version}

    def execute(self, action: ActionEvent, *, permissions: Iterable[str], replay: bool = False) -> ActionResultEvent | None:
        self.event_store.add_events([action])
        try:
            spec = self._actions[action.action_function]
        except KeyError as error:
            raise KeyError(f"action function {action.action_function!r} is not allowlisted") from error
        granted = set(permissions)
        if action.required_permission != spec.permission or spec.permission not in granted:
            raise PermissionError(f"action {action.action_function!r} requires permission {spec.permission!r}")
        if spec.argument_schema_ref:
            if self.schemas is None:
                raise ValueError("action argument schema is configured but no SchemaRegistry is available")
            self.schemas.validate(spec.argument_schema_ref, dict(action.arguments))

        state_key = f"action:{action.event_id}"
        prior = self.state.get("execution", state_key, {})
        declared_key = action.idempotency_key or action.event_id
        identity = {
            "function": action.action_function,
            "permission": action.required_permission,
            "arguments": action.arguments,
            "argument_schema_ref": spec.argument_schema_ref,
            "result_schema_ref": spec.result_schema_ref,
            "pass_idempotency_key": spec.pass_idempotency_key,
            "retry_limit": action.retry_limit,
        }
        request_hash = sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
        dedupe_key = sha256(f"{action.action_function}\0{declared_key}".encode()).hexdigest()
        if replay:
            saved_action = self.state.action_idempotency(dedupe_key, request_hash)
            if saved_action:
                return self._existing_action_result(saved_action, dedupe_key)
            if prior.get("status") in {"succeeded", "failed"}:
                return self._find_result(prior.get("result_event_id"))
            self.state.set("execution", state_key, {
                "action_event_id": action.event_id, "idempotency_key": declared_key,
                "status": "skipped_replay", "attempts": prior.get("attempts", 0), "result_event_id": None,
            }, "external action skipped during replay")
            return None

        legacy_record = prior
        if prior.get("status") == "failed" and int(prior.get("attempts", 0)) < action.retry_limit + 1:
            legacy_record = {**prior, "status": "retrying"}
        owner, existing = self.state.claim_action(
            dedupe_key, request_hash, action.event_id, legacy_record=legacy_record,
        )
        if not owner:
            outcome = self._existing_action_result(existing, dedupe_key)
            if outcome is not None or existing["status"] in {"succeeded", "failed"}:
                return outcome
            # Another executor has reserved this key. Returning no outcome
            # prevents a duplicate side effect while that request completes.
            return None

        last_result: ActionResultEvent | None = None
        for attempt in range(int(existing["attempts"]) + 1, action.retry_limit + 2):
            try:
                if spec.pass_idempotency_key:
                    result = spec.function(dict(action.arguments), idempotency_key=declared_key) or {}
                else:
                    result = spec.function(dict(action.arguments)) or {}
                if not isinstance(result, dict):
                    raise ValueError("action functions must return a structured object")
                if spec.result_schema_ref:
                    if self.schemas is None:
                        raise ValueError("action result schema is configured but no SchemaRegistry is available")
                    self.schemas.validate(spec.result_schema_ref, dict(result))
            except Exception as error:  # action failures are durable outcomes, not silent exceptions
                last_result = self._result_event(action, attempt, "failed", {"error": str(error)})
                self.event_store.add_events([last_result])
                self.state.set("execution", state_key, {
                    "action_event_id": action.event_id,
                    "idempotency_key": declared_key,
                    "status": "failed", "attempts": attempt,
                    "result_event_id": last_result.event_id,
                }, f"action attempt {attempt} failed")
                if attempt >= action.retry_limit + 1:
                    self.state.finish_action(dedupe_key, status="failed", attempts=attempt,
                                             result_event_id=last_result.event_id)
                    return last_result
                continue

            # Keep persistence errors outside the action-function exception
            # handler: a successful side effect must never be mislabeled as a
            # failed action just because recording its outcome failed.
            last_result = self._result_event(action, attempt, "succeeded", result)
            self.event_store.add_events([last_result])
            self.state.set("execution", state_key, {
                "action_event_id": action.event_id,
                "idempotency_key": declared_key,
                "status": "succeeded", "attempts": attempt,
                "result_event_id": last_result.event_id,
            }, "action completed")
            self.state.finish_action(dedupe_key, status="succeeded", attempts=attempt,
                                     result_event_id=last_result.event_id)
            return last_result
        return last_result

    def _result_event(self, action: ActionEvent, attempt: int, status: Literal["succeeded", "failed"],
                      outcome: dict[str, Any]) -> ActionResultEvent:
        if self.schemas is not None:
            self.schemas.validate("reflex.action-result@1.0", outcome)
        return ActionResultEvent(
            event_id=stable_id("reflex-action-result", action.event_id, str(attempt), status),
            event_type=f"action.{status}", event_time=datetime.now(timezone.utc),
            entity_ids=action.entity_ids, source=f"executor:{action.action_function}",
            provenance={"action_function": action.action_function, "action_event_id": action.event_id},
            parent_event_ids=(action.event_id,), correlation_id=action.correlation_id,
            causation_id=action.event_id, schema_ref="reflex.action-result@1.0",
            idempotency_key=f"{action.idempotency_key or action.event_id}:{attempt}:{status}",
            access_scope=action.access_scope, payload=outcome,
            action_event_id=action.event_id, status=status, attempt=attempt,
        )

    def _find_result(self, event_id: str | None) -> ActionResultEvent | None:
        if not event_id:
            return None
        for event in self.event_store.events():
            if event.event_id == event_id and isinstance(event, ActionResultEvent):
                return event
        return None

    def _existing_action_result(self, record: dict[str, Any], dedupe_key: str) -> ActionResultEvent | None:
        """Return a prior result and recover a committed result after a crash."""
        if record["status"] in {"succeeded", "failed"}:
            return self._find_result(record.get("result_event_id"))
        if record["status"] != "running":
            return None
        events = self.event_store.events()
        action = next((event for event in events
                       if isinstance(event, ActionEvent) and event.event_id == record["action_event_id"]), None)
        outcomes = [event for event in events if isinstance(event, ActionResultEvent)
                    and event.action_event_id == record["action_event_id"]]
        if action is None or not outcomes:
            return None
        outcome = max(outcomes, key=lambda event: event.attempt)
        # A success is terminal at any attempt. A failure can be terminal only
        # after the action's declared retry allowance has been exhausted.
        if outcome.status != "succeeded" and outcome.attempt < action.retry_limit + 1:
            return None
        self.state.finish_action(dedupe_key, status=outcome.status, attempts=outcome.attempt,
                                 result_event_id=outcome.event_id)
        self.state.set("execution", f"action:{action.event_id}", {
            "action_event_id": action.event_id,
            "idempotency_key": action.idempotency_key or action.event_id,
            "status": outcome.status, "attempts": outcome.attempt,
            "result_event_id": outcome.event_id,
        }, "recovered action outcome from event ledger")
        return outcome


class ReactionRun(FrozenModel):
    run_id: str
    reaction_id: str
    reaction_version: str
    trigger_event_id: str
    status: Literal["succeeded", "failed", "budget_exceeded", "skipped"]
    emitted_event_ids: tuple[str, ...] = ()
    nodes_executed: int = 0
    function_versions: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    replay: bool = False
    skip_reason: str | None = None
    state_snapshot: dict[str, Any] = Field(default_factory=dict)
    state_snapshot_source: Literal["captured", "provided", "live_run"] = "captured"
    state_snapshot_hash: str
    function_state_snapshots: dict[str, dict[str, Any]] = Field(default_factory=dict)
    function_state_snapshots_hash: str = ""
    function_providers: dict[str, dict[str, str | None]] = Field(default_factory=dict)
    started_at: datetime
    finished_at: datetime


class ReactionRuntime:
    """Run versioned event workflows and journal each decision for replay."""

    def __init__(self, event_store: EventStore, state: StateStore, functions: FunctionRegistry,
                 schemas: SchemaRegistry | None = None, actions: ActionExecutor | None = None):
        self.event_store = event_store
        self.state = state
        self.functions = functions
        self.schemas = schemas or SchemaRegistry(state)
        self.actions = actions

    def register_definition(self, definition: ReactionDefinition) -> None:
        key = f"definition:{definition.reaction_id}@{definition.version}"
        record = definition.model_dump(mode="json")
        prior = self.state.get("execution", key)
        if prior is None:
            self.state.set("execution", key, record,
                           f"register Reaction {definition.reaction_id}@{definition.version}")
        elif prior != record:
            raise ValueError(f"Reaction {definition.reaction_id}@{definition.version} is already registered with different content")

    def load_definition(self, reaction_id: str, version: str) -> ReactionDefinition:
        key = f"definition:{reaction_id}@{version}"
        record = self.state.get("execution", key)
        if record is None:
            raise KeyError(f"unknown Reaction {reaction_id}@{version}")
        return ReactionDefinition.model_validate(record)

    def process(self, event: Event, definition: ReactionDefinition, *, replay: bool = False,
                state_snapshot: dict[str, Any] | None = None) -> ReactionRun:
        """Match one input, validate its scope, run registered nodes, and save outputs.

        A replay records the same computation but asks the action executor to skip
        external side effects. The returned run links the trigger, function versions,
        state snapshot, and emitted event IDs for review.
        """
        self.register_definition(definition)
        self.event_store.add_events([event])
        if not self._matches(definition.subscriptions, event.event_type):
            return self._record_empty(event, definition, replay, "not subscribed")
        if event.access_scope and not set(event.access_scope).issubset(set(definition.read_scopes)):
            return self._record_empty(event, definition, replay,
                                      "Reaction is not granted the event access scope")
        run_id = stable_id("reflex-run", definition.reaction_id, definition.version, event.event_id,
                           "replay" if replay else "live")
        run_key = f"run:{run_id}"
        saved = self.state.get("execution", run_key)
        if saved and saved.get("status") in {"succeeded", "failed", "budget_exceeded"}:
            return ReactionRun.model_validate(saved)

        # Resolve even downstream dependencies before a node can write state or request an action.
        # A missing library can be loaded and the same trigger submitted again.
        bound = self.functions.bind(definition)
        started = datetime.now(timezone.utc)
        replay_function_snapshots: dict[str, dict[str, Any]] = {}
        if state_snapshot is not None:
            pinned_state = deepcopy(state_snapshot)
            snapshot_source: Literal["captured", "provided", "live_run"] = "provided"
        elif replay:
            live_run_id = stable_id("reflex-run", definition.reaction_id, definition.version,
                                    event.event_id, "live")
            live_run = self.state.get("execution", f"run:{live_run_id}")
            if live_run is not None:
                saved_live_run = ReactionRun.model_validate(live_run)
                pinned_state = deepcopy(saved_live_run.state_snapshot)
                replay_function_snapshots = deepcopy(saved_live_run.function_state_snapshots)
                snapshot_source = "live_run"
            else:
                pinned_state = self._capture_state(definition)
                snapshot_source = "captured"
        else:
            pinned_state = self._capture_state(definition)
            snapshot_source = "captured"
        state_hash = sha256(json.dumps(pinned_state, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
        node_by_id = {node.node_id: node for node in definition.nodes}
        entries = tuple(node.node_id for node in definition.nodes
                        if node.accepts_event_types and self._matches(node.accepts_event_types, event.event_type))
        if not entries:
            entries = definition.resolved_entry_nodes()
        queue = deque((node_id, event) for node_id in entries)
        visited: set[tuple[str, str]] = set()
        emitted: dict[str, Event] = {}
        errors: list[str] = []
        versions: set[str] = set()
        function_state_snapshots: dict[str, dict[str, Any]] = {}
        executed = 0
        budget_exceeded = False

        missing_definition_state = [requirement for requirement in definition.state_requirements
                                   if not self._pinned_state_requirement(requirement, pinned_state)]
        if missing_definition_state:
            errors.append("missing Reaction state requirement(s): " + ", ".join(missing_definition_state))
            queue.clear()

        def condition_matches(ref: str | None, item: Event, invocation: str) -> bool:
            if ref is None:
                return True
            versions.add(ref)
            snapshot = invocation_snapshot(invocation)
            try:
                return bool(bound[ref].function(item, snapshot))
            except Exception as error:
                errors.append(f"condition {ref} on {item.event_id}: {error}")
                return False

        def invocation_snapshot(invocation: str) -> dict[str, Any]:
            if replay and state_snapshot is None and invocation in replay_function_snapshots:
                snapshot = deepcopy(replay_function_snapshots[invocation])
            else:
                snapshot = self._merged_snapshot(pinned_state)
            function_state_snapshots[invocation] = deepcopy(snapshot)
            return snapshot

        while queue:
            node_id, current = queue.popleft()
            node = node_by_id[node_id]
            visit_key = (node_id, current.event_id)
            if visit_key in visited:
                continue
            if executed >= definition.max_steps or len(emitted) >= definition.max_emitted_events:
                budget_exceeded = True
                break
            visited.add(visit_key)
            if node.accepts_event_types and not self._matches(node.accepts_event_types, current.event_type):
                continue
            if not condition_matches(node.condition_ref, current,
                                     f"condition:{node.node_id}:{current.event_id}"):
                continue
            spec = bound[node.function.ref]
            versions.add(spec.ref.ref)
            if node.input_schema_refs:
                if current.schema_ref not in node.input_schema_refs:
                    errors.append(f"node {node.node_id}: input schema {current.schema_ref!r} is not declared")
                    continue
                try:
                    self.schemas.validate(current.schema_ref, dict(current.payload))
                except (KeyError, ValueError) as error:
                    errors.append(f"node {node.node_id}: input contract failed: {error}")
                    continue
            invocation = f"function:{spec.ref.ref}:{node.node_id}:{current.event_id}"
            snapshot = invocation_snapshot(invocation)
            if any(not self._state_requirement(requirement, snapshot) for requirement in node.state_requirements):
                errors.append(f"node {node.node_id}: missing state requirement")
                continue
            function_outputs: list[Event] = []
            for attempt in range(definition.max_retries + 1):
                try:
                    function_outputs = list(spec.function(current, deepcopy(snapshot)))
                    break
                except Exception as error:
                    if attempt == definition.max_retries:
                        errors.append(f"node {node.node_id} ({spec.ref.ref}): {error}")

            executed += 1
            for produced in function_outputs:
                if len(emitted) >= definition.max_emitted_events:
                    budget_exceeded = True
                    break
                if not isinstance(produced, Event):
                    errors.append(f"node {node.node_id}: returned a non-Event output")
                    continue
                if node.output_schema_refs and produced.schema_ref not in node.output_schema_refs:
                    errors.append(f"node {node.node_id}: output schema {produced.schema_ref!r} is not declared")
                    continue
                if produced.schema_ref:
                    try:
                        self.schemas.validate(produced.schema_ref, dict(produced.payload))
                    except (KeyError, ValueError) as error:
                        errors.append(f"node {node.node_id}: output contract failed: {error}")
                        continue
                produced = self._attach_lineage(produced, current, definition, node, spec.ref.ref)
                try:
                    self.event_store.add_events([produced])
                except ValueError as error:
                    errors.append(f"node {node.node_id}: {error}")
                    continue
                emitted[produced.event_id] = produced
                downstream_events = [produced]
                if isinstance(produced, ActionEvent):
                    versions.add(produced.action_function)
                    if not definition.permissions:
                        errors.append(f"node {node.node_id}: Action Event produced by a Reaction with no permissions")
                    elif self.actions is None:
                        errors.append(f"node {node.node_id}: no action executor is configured")
                    elif len(emitted) + produced.retry_limit + 1 > definition.max_emitted_events:
                        budget_exceeded = True
                        errors.append(f"node {node.node_id}: insufficient output budget for action attempts")
                    else:
                        try:
                            outcome = self.actions.execute(produced, permissions=definition.permissions, replay=replay)
                            if not replay:
                                outcomes = self._action_outcomes(produced.event_id)
                                if outcome is not None and all(item.event_id != outcome.event_id for item in outcomes):
                                    outcomes.append(outcome)
                                for result_event in outcomes:
                                    emitted[result_event.event_id] = result_event
                                    downstream_events.append(result_event)
                            if outcome is not None and outcome.status == "failed":
                                errors.append(f"node {node.node_id}: action {produced.event_id} failed after retries")
                        except (KeyError, PermissionError, ValueError) as error:
                            errors.append(f"node {node.node_id}: action execution rejected: {error}")
                for output in downstream_events:
                    for edge_index, edge in enumerate(node.edges):
                        if edge.event_types and not self._matches(edge.event_types, output.event_type):
                            continue
                        edge_invocation = f"edge:{node.node_id}:{edge_index}:{output.event_id}"
                        if condition_matches(edge.condition_ref, output,
                                             f"condition:{edge.condition_ref}:{edge_invocation}"):
                            queue.append((edge.target, output))
                    if self._matches(definition.subscriptions, output.event_type):
                        for entry_node in self._entry_nodes(node_by_id, definition, output.event_type):
                            queue.append((entry_node, output))
            if budget_exceeded:
                break

        status: Literal["succeeded", "failed", "budget_exceeded"] = (
            "budget_exceeded" if budget_exceeded else "failed" if errors else "succeeded"
        )
        function_providers: dict[str, dict[str, str | None]] = {}
        for function_ref in versions:
            if function_ref in bound:
                spec = bound[function_ref]
                function_providers[function_ref] = {
                    "provider": spec.provider,
                    "provider_version": spec.provider_version,
                }
            elif self.actions is not None:
                details = self.actions.provider_details(function_ref)
                if details is not None:
                    function_providers[function_ref] = details
        function_snapshots_hash = sha256(json.dumps(
            function_state_snapshots, sort_keys=True, separators=(",", ":"), default=str
        ).encode()).hexdigest()
        result = ReactionRun(
            run_id=run_id, reaction_id=definition.reaction_id, reaction_version=definition.version,
            trigger_event_id=event.event_id, status=status, emitted_event_ids=tuple(emitted),
            nodes_executed=executed, function_versions=tuple(sorted(versions)), errors=tuple(errors),
            replay=replay, state_snapshot=pinned_state, state_snapshot_source=snapshot_source,
            state_snapshot_hash=state_hash, function_state_snapshots=function_state_snapshots,
            function_state_snapshots_hash=function_snapshots_hash,
            function_providers=function_providers,
            started_at=started, finished_at=datetime.now(timezone.utc),
        )
        self.state.set("execution", run_key, result.model_dump(mode="json"),
                       f"Reaction {definition.reaction_id}@{definition.version} {status}")
        return result

    def _pinned_state_requirement(self, requirement: str, pinned_state: dict[str, Any]) -> bool:
        if ":" in requirement:
            _namespace, key = requirement.split(":", 1)
            value = pinned_state.get(requirement, pinned_state.get(key))
            return value is not None
        return bool(pinned_state.get(requirement))

    @staticmethod
    def _state_requirement(requirement: str, snapshot: dict[str, Any]) -> bool:
        if ":" in requirement:
            return requirement in snapshot and snapshot[requirement] is not None
        return bool(snapshot.get(requirement)) or any(key.startswith(f"{requirement}:") for key in snapshot)

    def _capture_state(self, definition: ReactionDefinition) -> dict[str, Any]:
        captured: dict[str, Any] = {}
        requirements = set(definition.state_requirements)
        requirements.update(requirement for node in definition.nodes for requirement in node.state_requirements)
        for requirement in sorted(requirements):
            if ":" in requirement:
                namespace, key = requirement.split(":", 1)
                value = self.state.get(namespace, key, None)
                if value is not None:
                    captured[requirement] = value
            else:
                captured[requirement] = self.state.snapshot(requirement)
        return captured

    def _merged_snapshot(self, pinned_state: dict[str, Any]) -> dict[str, Any]:
        # Preserve the legacy flat lookup while also recording qualified names.
        # Per-invocation snapshots make changes between nodes reproducible.
        # Run records contain their own input snapshots. Re-introducing that
        # execution namespace here would nest every past trace into the next
        # one and make durable snapshots grow exponentially.
        snapshot: dict[str, Any] = self.state.snapshot_context(exclude_namespaces={"execution"})
        for requirement, value in pinned_state.items():
            if ":" in requirement:
                _namespace, key = requirement.split(":", 1)
                snapshot[requirement] = deepcopy(value)
                snapshot[key] = deepcopy(value)
            elif isinstance(value, dict):
                snapshot[requirement] = deepcopy(value)
                snapshot.update(deepcopy(value))
            else:
                snapshot[requirement] = deepcopy(value)
        return snapshot

    @staticmethod
    def _entry_nodes(node_by_id: dict[str, ReactionNode], definition: ReactionDefinition,
                     event_type: str) -> tuple[str, ...]:
        matching = tuple(node.node_id for node in node_by_id.values()
                         if node.accepts_event_types and ReactionRuntime._matches(node.accepts_event_types, event_type))
        return matching or definition.resolved_entry_nodes()

    def _action_outcomes(self, action_event_id: str) -> list[ActionResultEvent]:
        return [event for event in self.event_store.events()
                if isinstance(event, ActionResultEvent) and event.action_event_id == action_event_id]

    @staticmethod
    def _matches(patterns: Iterable[str], event_type: str) -> bool:
        return any(fnmatchcase(event_type, pattern) for pattern in patterns)

    @staticmethod
    def _attach_lineage(output: Event, parent: Event, definition: ReactionDefinition,
                        node: ReactionNode, function_ref: str) -> Event:
        provenance = dict(output.provenance)
        provenance.update({"reaction": f"{definition.reaction_id}@{definition.version}",
                           "node": node.node_id, "function": function_ref})
        return output.model_copy(update={
            "parent_event_ids": output.parent_event_ids or (parent.event_id,),
            "correlation_id": output.correlation_id or parent.correlation_id or parent.event_id,
            "causation_id": output.causation_id or parent.event_id,
            "access_scope": output.access_scope or parent.access_scope,
            "provenance": provenance,
        })

    def _record_empty(self, event: Event, definition: ReactionDefinition, replay: bool, reason: str) -> ReactionRun:
        run_id = stable_id("reflex-run", definition.reaction_id, definition.version, event.event_id,
                           "replay" if replay else "live")
        prior = self.state.get("execution", f"run:{run_id}")
        if prior:
            return ReactionRun.model_validate(prior)
        now = datetime.now(timezone.utc)
        result = ReactionRun(run_id=run_id,
                             reaction_id=definition.reaction_id, reaction_version=definition.version,
                             trigger_event_id=event.event_id, status="skipped", errors=(), replay=replay,
                             skip_reason=reason, state_snapshot={}, state_snapshot_hash=sha256(b"{}").hexdigest(),
                             started_at=now, finished_at=now)
        self.state.set("execution", f"run:{result.run_id}", result.model_dump(mode="json"),
                       f"Reaction {definition.reaction_id}@{definition.version} skipped: {reason}")
        return result


__all__ = ["FunctionRegistry", "FunctionSpec", "ActionExecutor", "ReactionRun", "ReactionRuntime"]
