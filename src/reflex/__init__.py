"""REFLEX event-driven intelligence and discovery framework."""

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
from .ingestion import GenericNewsAdapter, NewsStoryAdapter

__version__ = "0.2.0"

__all__ = [
    "Event", "NativeEvent", "ObservationEvent", "ActionEvent", "ActionResultEvent",
    "ExploratoryObservation", "FunctionRef", "ReactionEdge", "ReactionNode", "ReactionDefinition",
    "SchemaRegistry", "FunctionRegistry", "ActionExecutor", "ReactionRun", "ReactionRuntime",
    "DetectorPlugin", "DetectorRegistry",
    "GenericNewsAdapter", "NewsStoryAdapter",
]
