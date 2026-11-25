"""Execution-environment interfaces; mock classes are intentionally marked mock."""

from __future__ import annotations

from typing import Literal, Protocol

from .models.inputs import AgentInput, RunBudget
from .models.results import AdapterOutput


class ElizaHarness(Protocol):
    """Minimal port for an Eliza runtime to implement when one is available."""
    execution_kind: Literal["external", "mock"]

    def execute_workflow(self, experiment: AgentInput, *, workflow: str,
                         budgets: RunBudget, trial_seed: int) -> AdapterOutput: ...


class MockElizaHarness:
    """Contract-checking fixture only; it is not Eliza and proves no integration."""
    execution_kind = "mock"

    def execute_workflow(self, experiment: AgentInput, *, workflow: str,
                         budgets: RunBudget, trial_seed: int) -> AdapterOutput:
        return AdapterOutput(compute_seconds=0.0, metadata={
            "harness": "mock", "workflow": workflow, "trial_seed": trial_seed,
            "input_hash": experiment.input_hash,
            "integration_claim": "not_eliza_integration",
        })


class ElizaExperimentAdapter:
    """Bridge an Eliza-compatible harness into the shared REFLEX runner."""
    variant_id = "G"

    def __init__(self, harness: ElizaHarness, *, workflow: str = "long_running_event_workflow"):
        self.harness = harness
        self.workflow = workflow
        self.execution_kind = harness.execution_kind

    def run(self, experiment: AgentInput, budgets: RunBudget, *, trial_seed: int) -> AdapterOutput:
        return self.harness.execute_workflow(experiment, workflow=self.workflow,
                                             budgets=budgets, trial_seed=trial_seed)


__all__ = ["ElizaHarness", "MockElizaHarness", "ElizaExperimentAdapter"]
