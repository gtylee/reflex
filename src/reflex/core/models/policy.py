"""Policy decisions and evidence-backed investigation hypotheses."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Literal
from uuid import uuid4

from pydantic import Field

from .base import FrozenModel


class PolicyAction(StrEnum):
    """Allowed routes for a finding; values describe workflow intent."""

    RECORD = "record"
    INVESTIGATE = "investigate"
    DEFER = "defer"
    ESCALATE = "escalate"
    PROPOSE_LENS = "propose_lens"
    IGNORE = "ignore"


class PolicyDecision(FrozenModel):
    """Versioned record explaining the next workflow route for a finding.

    The decision captures policy output, not evidence validity. In particular,
    ``INVESTIGATE`` means gather more information, not execute an external
    action.
    """

    decision_id: str = Field(default_factory=lambda: str(uuid4()))
    observation_id: str
    action: PolicyAction
    reason: str
    policy_id: str
    policy_version: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Hypothesis(FrozenModel):
    """A tentative explanation with separately linked evidence on both sides.

    Confidence is an application-supplied value and is not calibrated by this
    record type. Use ``status`` and the evidence tuples to keep open ideas
    distinct from explanations later supported or contradicted by review.
    """

    hypothesis_id: str = Field(default_factory=lambda: str(uuid4()))
    statement: str
    status: Literal["open", "supported", "contradicted", "deferred", "rejected"] = "open"
    supporting_evidence: tuple[str, ...] = ()
    contradicting_evidence: tuple[str, ...] = ()
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
