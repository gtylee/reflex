"""Typed workflow graph definitions used by the REFLEX runtime."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from reflex.core.models.base import FrozenModel


class FunctionRef(FrozenModel):
    """A registered callable name and version, never executable code."""

    function_id: str
    version: str
    kind: Literal["lens", "policy", "skill", "condition"]

    @property
    def ref(self) -> str:
        return f"{self.function_id}@{self.version}"


class ReactionEdge(FrozenModel):
    """A typed connection from one Reaction node to another."""

    target: str
    event_types: tuple[str, ...] = ()
    condition_ref: str | None = None


class ReactionNode(FrozenModel):
    """One function and its input, output, state, and transition contracts."""

    node_id: str
    function: FunctionRef
    accepts_event_types: tuple[str, ...] = ()
    input_schema_refs: tuple[str, ...] = ()
    output_schema_refs: tuple[str, ...] = ()
    state_requirements: tuple[str, ...] = ()
    condition_ref: str | None = None
    edges: tuple[ReactionEdge, ...] = ()


class ReactionDefinition(FrozenModel):
    """A versioned workflow graph with declared entry points and run limits."""

    reaction_id: str
    version: str
    subscriptions: tuple[str, ...]
    nodes: tuple[ReactionNode, ...]
    entry_nodes: tuple[str, ...] = ()
    read_scopes: tuple[str, ...] = ()
    permissions: tuple[str, ...] = ()
    state_requirements: tuple[str, ...] = ()
    max_steps: int = Field(default=100, ge=1, le=1000)
    max_emitted_events: int = Field(default=100, ge=1, le=1000)
    max_retries: int = Field(default=0, ge=0, le=5)

    @model_validator(mode="after")
    def validate_graph(self) -> "ReactionDefinition":
        if not self.subscriptions:
            raise ValueError("a Reaction must subscribe to at least one event type")
        node_ids = [node.node_id for node in self.nodes]
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("Reaction node IDs must be unique")
        known = set(node_ids)
        if not known:
            raise ValueError("a Reaction needs at least one node")
        if any(node not in known for node in self.entry_nodes):
            raise ValueError("entry_nodes must refer to declared nodes")
        for node in self.nodes:
            for edge in node.edges:
                if edge.target not in known:
                    raise ValueError(f"edge from {node.node_id!r} refers to unknown node {edge.target!r}")
        if not self.entry_nodes and not (known - {edge.target for node in self.nodes for edge in node.edges}):
            raise ValueError("a cyclic Reaction graph must declare an entry node")
        return self

    def resolved_entry_nodes(self) -> tuple[str, ...]:
        if self.entry_nodes:
            return self.entry_nodes
        targets = {edge.target for node in self.nodes for edge in node.edges}
        return tuple(node.node_id for node in self.nodes if node.node_id not in targets)
