"""Typed adapter outputs and summarized experiment results."""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import Field

from reflex.core.models.base import FrozenModel

from .inputs import AgentInput, RunBudget


class AdapterOutput(FrozenModel):
    """Predictions, evidence references, actions, and resource usage from one adapter."""

    predicted_event_ids: tuple[str, ...] = ()
    evidence_refs: dict[str, tuple[str, ...]] = Field(default_factory=dict)
    discoveries: tuple[dict[str, Any], ...] = ()
    actions: tuple[dict[str, Any], ...] = ()
    token_count: int = Field(default=0, ge=0)
    compute_seconds: float = Field(default=0, ge=0)
    cost_usd: float = Field(default=0, ge=0)
    model_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    investigation_count: int = Field(default=0, ge=0)
    output_variance_value: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExperimentAdapter(Protocol):
    """Adapter contract; all baselines receive the same label-blind input bundle."""

    variant_id: str
    execution_kind: Literal["deterministic", "external", "mock"]

    def run(self, experiment: AgentInput, budgets: RunBudget, *, trial_seed: int) -> AdapterOutput: ...


class VariantRun(FrozenModel):
    """Outcome and metrics for one evaluated variant."""

    variant_id: str
    description: str
    execution_kind: Literal["deterministic", "external", "mock", "unclassified"] = "unclassified"
    status: Literal["completed", "adapter_unavailable", "failed", "over_budget"]
    input_hash: str
    metrics: dict[str, Any] | None = None
    output: AdapterOutput | None = None
    error: str | None = None


class ExperimentReport(FrozenModel):
    """Comparison-ready summary with explicit readiness and caveats."""

    config_id: str
    config_version: str
    scenario: str
    seed: int
    input_hash: str
    generated_event_count: int
    truth_event_count: int
    test_truth_event_count: int = 0
    runs: tuple[VariantRun, ...]
    comparison_ready: bool
    readiness_reasons: tuple[str, ...] = ()
    note: str = "Missing baseline adapters remain unrun; no performance claim is inferred."


VARIANT_DESCRIPTIONS: dict[str, str] = {
    "A": "Conventional LLM with direct access to underlying data",
    "B": "Conventional tool-enabled agent with analytical tools",
    "C": "Agent with predefined deterministic lenses",
    "D": "REFLEX without persistent state",
    "E": "REFLEX without curation",
    "F": "REFLEX with persistent state, deterministic reactions, skills, and curation",
    "G": "Paired conventional Eliza versus Eliza with REFLEX integration",
}
