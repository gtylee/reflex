"""Typed experiment settings and label-separated input records."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import Field

from reflex.core.models.base import FrozenModel
from reflex.core.models.events import Event


class RunBudget(FrozenModel):
    """Resource ceilings shared by one experiment run."""

    inference_tokens: int = Field(default=20_000, ge=0)
    compute_seconds: float = Field(default=900, gt=0)
    tool_calls: int = Field(default=100, ge=0)


class ExperimentConfig(FrozenModel):
    """Reproducible settings for one experiment family and scenario."""

    config_id: str
    version: str = "1.0"
    phase: int = Field(ge=1, le=6)
    family: str
    scenario: str
    objective: str
    seed: int = 11
    entity_count: int = Field(default=32, ge=4)
    days: int = Field(default=30, ge=10)
    budgets: RunBudget = Field(default_factory=RunBudget)
    variants: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G")
    parameters: dict[str, Any] = Field(default_factory=dict)


class GroundTruthLabel(FrozenModel):
    """Hidden evaluation label; it must not be passed to the evaluated adapter."""

    event_id: str
    entity_id: str
    mechanism: str
    event_time: datetime
    note: str = ""


class AgentInput(FrozenModel):
    """Only visible experiment inputs; hidden labels never cross this boundary."""

    case_id: str
    input_hash: str
    events: tuple[Event, ...]
    train_end: datetime
    calibration_end: datetime
    end: datetime
    objective: str
    input_schema_version: str = "1.0"
