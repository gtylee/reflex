"""Live registration of versioned Reaction functions supplied by the host or installed libraries."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import metadata
from threading import Lock
from typing import Any

from .contracts.reactions import FunctionRef, ReactionDefinition
from .core.models.events import Event


ReactionFunction = Callable[[Event, dict[str, dict[str, Any]]], Iterable[Event]]
ConditionFunction = Callable[[Event, dict[str, dict[str, Any]]], bool]
ENTRY_POINT_GROUP = "reflex_intelligence.functions"


@dataclass(frozen=True)
class FunctionSpec:
    ref: FunctionRef
    function: ReactionFunction | ConditionFunction
    description: str = ""
    deterministic: bool = True


class FunctionRegistry:
    """The host chooses providers. Reactions refer to their functions by exact version."""

    def __init__(self):
        self._functions: dict[str, FunctionSpec] = {}
        self._lock = Lock()

    @staticmethod
    def available_libraries() -> tuple[str, ...]:
        """Read installed package metadata without importing any provider."""
        return tuple(sorted({ep.name for ep in metadata.entry_points(group=ENTRY_POINT_GROUP)}))

    def register(self, ref: FunctionRef, function: ReactionFunction | ConditionFunction,
                 *, description: str = "", deterministic: bool = True) -> None:
        self.register_many([FunctionSpec(ref, function, description, deterministic)])

    def register_many(self, specs: Iterable[FunctionSpec]) -> None:
        """Publish the whole collection together, retaining previously registered versions."""
        entries = tuple(specs)
        with self._lock:
            updated = self._functions.copy()
            for spec in entries:
                if not isinstance(spec, FunctionSpec) or not isinstance(spec.ref, FunctionRef):
                    raise TypeError("expected a FunctionSpec containing a FunctionRef")
                if not callable(spec.function):
                    raise TypeError("expected a FunctionSpec containing a FunctionRef and a callable")
                prior = updated.get(spec.ref.ref)
                if prior is not None and prior != spec:
                    raise ValueError(f"function {spec.ref.ref!r} is already registered")
                updated[spec.ref.ref] = spec
            self._functions = updated

    def load_library(self, name: str) -> tuple[FunctionRef, ...]:
        """Load one installed provider chosen by the host. It returns an iterable of specs."""
        matches = tuple(metadata.entry_points(group=ENTRY_POINT_GROUP, name=name))
        if not matches:
            raise KeyError(f"no installed REFLEX function library named {name!r}")
        if len(matches) != 1:
            raise ValueError(f"multiple installed libraries claim {name!r}")
        provider = matches[0].load()
        specs = tuple(provider())
        self.register_many(specs)
        return tuple(spec.ref for spec in specs)

    def resolve(self, ref: FunctionRef) -> FunctionSpec:
        with self._lock:
            return self._resolve(self._functions, ref.ref, ref.kind)

    @staticmethod
    def _resolve(functions: dict[str, FunctionSpec], key: str, kind: str) -> FunctionSpec:
        try:
            spec = functions[key]
        except KeyError as error:
            raise KeyError(f"unregistered function {key!r}") from error
        if spec.ref.kind != kind:
            raise ValueError(f"function kind mismatch for {key!r}: expected {kind}")
        return spec

    def bind(self, definition: ReactionDefinition) -> dict[str, FunctionSpec]:
        """Resolve every node and condition against the same published set before execution."""
        with self._lock:
            functions = self._functions
        bound = {}
        for node in definition.nodes:
            ref = node.function
            if ref.kind == "condition":
                raise ValueError(f"node {node.node_id!r} requires an event-producing function")
            bound[ref.ref] = self._resolve(functions, ref.ref, ref.kind)
            for condition in (node.condition_ref, *(edge.condition_ref for edge in node.edges)):
                if condition is not None:
                    bound[condition] = self._resolve(functions, condition, "condition")
        return bound
