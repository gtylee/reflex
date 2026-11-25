"""Live registration of versioned Reaction functions supplied by the host or installed libraries."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from importlib import metadata
from math import isfinite
from threading import Lock
from typing import Any

from .contracts.reactions import FunctionRef, ReactionDefinition
from .core.models.events import Event


ReactionFunction = Callable[[Event, dict[str, Any]], Iterable[Event]]
ConditionFunction = Callable[[Event, dict[str, Any]], bool]
ENTRY_POINT_GROUP = "reflex_intelligence.functions"
DETECTOR_ENTRY_POINT_GROUP = "reflex_intelligence.detectors"


@dataclass(frozen=True)
class FunctionSpec:
    ref: FunctionRef
    function: ReactionFunction | ConditionFunction
    description: str = ""
    deterministic: bool = True
    provider: str | None = None
    provider_version: str | None = None


@dataclass(frozen=True)
class DetectorPlugin:
    """A training-only detector offered to REFLEX's statistical lens selector.

    ``match_score`` and ``fit`` receive only the training profile/events.
    ``score`` is called after a detector is selected and must return a score
    for every event in the supplied sequence, keyed by event ID.
    """

    lens_id: str
    version: str
    description: str
    match_score: Callable[[dict[str, Any]], float | None]
    fit: Callable[[Sequence[Event], dict[str, Any]], Any]
    score: Callable[[Any, Sequence[Event]], Mapping[str, float]]
    features: tuple[str, ...] = ()
    provider: str | None = None
    provider_version: str | None = None

    @property
    def ref(self) -> str:
        return f"{self.lens_id}@{self.version}"


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
                 *, description: str = "", deterministic: bool = True,
                 provider: str | None = None, provider_version: str | None = None) -> None:
        self.register_many([FunctionSpec(ref, function, description, deterministic,
                                         provider, provider_version)])

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
        entry_point = matches[0]
        provider = entry_point.load()
        package_name = entry_point.dist.metadata.get("Name") if entry_point.dist else name
        package_version = entry_point.dist.version if entry_point.dist else None
        specs = tuple(replace(spec, provider=package_name, provider_version=package_version)
                      for spec in provider())
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


class DetectorRegistry:
    """Registry for optional statistical detector libraries.

    Libraries are discovered from installed Python entry points when selection
    runs. The selector evaluates each detector's match score using a profile
    built from the training partition, then fits only the winning detector on
    that same partition.
    """

    def __init__(self):
        self._detectors: dict[str, DetectorPlugin] = {}
        self._lock = Lock()

    @staticmethod
    def available_libraries() -> tuple[str, ...]:
        """Return installed detector entry-point names without importing them."""
        return tuple(sorted({ep.name for ep in metadata.entry_points(group=DETECTOR_ENTRY_POINT_GROUP)}))

    def register(self, detector: DetectorPlugin) -> None:
        """Add one in-process detector to the live selector registry."""
        self.register_many([detector])

    def register_many(self, detectors: Iterable[DetectorPlugin]) -> None:
        """Publish detector specs atomically and reject conflicting versions."""
        entries = tuple(detectors)
        with self._lock:
            updated = self._detectors.copy()
            for detector in entries:
                if not isinstance(detector, DetectorPlugin):
                    raise TypeError("expected DetectorPlugin entries")
                if not detector.lens_id or not detector.version:
                    raise ValueError("detector lens_id and version must be non-empty")
                if not callable(detector.match_score) or not callable(detector.fit) or not callable(detector.score):
                    raise TypeError("detector match_score, fit, and score must be callable")
                prior = updated.get(detector.ref)
                if prior is not None and prior != detector:
                    raise ValueError(f"detector {detector.ref!r} is already registered")
                updated[detector.ref] = detector
            self._detectors = updated

    def load_library(self, name: str) -> tuple[str, ...]:
        """Import a selected installed library and register its detector specs."""
        matches = tuple(metadata.entry_points(group=DETECTOR_ENTRY_POINT_GROUP, name=name))
        if not matches:
            raise KeyError(f"no installed REFLEX detector library named {name!r}")
        if len(matches) != 1:
            raise ValueError(f"multiple installed detector libraries claim {name!r}")
        entry_point = matches[0]
        package_name = entry_point.dist.metadata.get("Name") if entry_point.dist else name
        package_version = entry_point.dist.version if entry_point.dist else None
        provider = entry_point.load()
        detectors = tuple(replace(detector, provider=package_name, provider_version=package_version)
                          for detector in provider())
        self.register_many(detectors)
        return tuple(detector.ref for detector in detectors)

    def candidates(self, training_profile: dict[str, Any]) -> tuple[tuple[float, DetectorPlugin], ...]:
        """Return supported detectors sorted by descending training-only fit."""
        with self._lock:
            detectors = tuple(self._detectors.values())
        candidates = []
        for detector in detectors:
            score = detector.match_score(deepcopy(training_profile))
            if score is None:
                continue
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not isfinite(float(score)):
                raise ValueError(f"detector {detector.ref!r} returned an invalid match score")
            if not 0.0 <= float(score) <= 1.0:
                raise ValueError(f"detector {detector.ref!r} match score must be between 0 and 1")
            if score > 0:
                candidates.append((float(score), detector))
        return tuple(sorted(candidates, key=lambda item: (-item[0], item[1].ref)))
