"""REFLEX event-driven intelligence and discovery framework."""

from importlib import import_module

from reflex.core.models.events import ActionEvent, ActionResultEvent, Event, NativeEvent, ObservationEvent

from .contracts import (
    ExploratoryObservation,
    FunctionRef,
    ReactionDefinition,
    ReactionEdge,
    ReactionNode,
    SchemaRegistry,
)
from .runtime import ActionExecutor, FunctionRegistry, ReactionRun, ReactionRuntime
from .registry import DetectorPlugin, DetectorRegistry

__version__ = "0.2.0"

__all__ = [
    "Event", "NativeEvent", "ObservationEvent", "ActionEvent", "ActionResultEvent",
    "ExploratoryObservation", "FunctionRef", "ReactionEdge", "ReactionNode", "ReactionDefinition",
    "SchemaRegistry", "FunctionRegistry", "ActionExecutor", "ReactionRun", "ReactionRuntime",
    "DetectorPlugin", "DetectorRegistry",
    "GenericNewsAdapter", "NewsStoryAdapter",
]


def __getattr__(name: str):
    """Load legacy dataset adapters only when a caller explicitly requests them."""
    if name not in {"GenericNewsAdapter", "NewsStoryAdapter"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(".ingestion", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
