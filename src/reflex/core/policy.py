"""Deterministic decision routing and a fixed local investigation interface.

Policy decisions and tool calls are persisted or traced for review. This
module does not contain a language model and does not perform external actions.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Any, Callable
from uuid import uuid4

from .models.lenses import CandidateStatus, LensCandidate, Observation
from .models.policy import Hypothesis, PolicyAction, PolicyDecision
from .storage import EventStore, StateStore


class PolicyEngine:
    """Route observations using a small deterministic rule and run budgets."""

    def __init__(self, state: StateStore, *, run_id: str, max_investigations: int = 8, max_queries: int = 40):
        """Bind policy to a persisted run budget.

        Budget initialization is idempotent for a run ID. Reusing a run ID
        therefore resumes the original limits and usage instead of resetting
        the allowance.
        """
        self.state = state
        self.run_id = run_id
        self.state.initialize_budget(run_id, max_investigations, max_queries)
        self.policy_id = "deterministic.discovery"
        self.version = "1.0"

    def decide(self, observation: Observation) -> PolicyDecision:
        """Record whether to keep, investigate, or defer one observation.

        The decision trusts only the observation's ``is_anomaly`` flag. It
        does not independently validate the score or approve a business
        action. A positive flag consumes one investigation slot immediately.
        """
        flagged = bool(isinstance(observation.value, dict) and observation.value.get("is_anomaly"))
        if not flagged:
            action, reason = PolicyAction.RECORD, "Observation retained; calibrated threshold not crossed."
        elif self.state.consume_budget(self.run_id, "investigation"):
            action, reason = PolicyAction.INVESTIGATE, "Calibrated threshold crossed and investigation budget is available."
        else:
            action, reason = PolicyAction.DEFER, "Investigation budget exhausted; observation remains recorded."
        return PolicyDecision(observation_id=observation.observation_id, action=action, reason=reason,
                              policy_id=self.policy_id, policy_version=self.version)


class BoundedTools:
    """Fixed local investigation tools with a persisted per-run query allowance.

    Each public operation records its name, input summary, and timestamp in
    ``trace`` and consumes one query unit. These are data-access helpers, not a
    general Python execution facility or external-service connector.
    """

    def __init__(self, event_store: EventStore, state: StateStore, *, run_id: str, query_limit: int = 500):
        """Bind tools to local stores, a run budget, and a per-query row cap."""
        self.event_store, self.state, self.run_id, self.query_limit = event_store, state, run_id, query_limit
        self.trace: list[dict[str, Any]] = []

    def _query(self, action: str, details: dict[str, Any]) -> None:
        """Charge one query and append its input metadata to the local trace."""
        if not self.state.consume_budget(self.run_id, "query"):
            raise RuntimeError("investigation query budget exhausted")
        self.trace.append({"tool": action, "at": datetime.now(timezone.utc).isoformat(), "input": details})

    def inspect_observations(self, observations: list[Observation], *, limit: int = 20) -> list[dict[str, Any]]:
        """Return at most 100 serialized observations from the supplied batch."""
        self._query("inspect_observations", {"limit": limit})
        return [obs.model_dump(mode="json") for obs in observations[:max(0, min(limit, 100))]]

    def query_events(self, *, start: datetime | None = None, end: datetime | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Read a bounded, time-filtered set of native events from the ledger."""
        bounded = max(0, min(limit, self.query_limit))
        self._query("query_events", {"start": start, "end": end, "limit": bounded})
        return [event.model_dump(mode="json") for event in self.event_store.native_events(start, end, bounded)]

    def compare_cohorts(self, observations: list[Observation], *, feature: str) -> dict[str, Any]:
        """Summarize one available numeric feature by observation cohort."""
        self._query("compare_cohorts", {"feature": feature})
        groups: dict[str, list[float]] = {}
        for observation in observations:
            features = observation.value.get("features", {}) if isinstance(observation.value, dict) else {}
            cohort = str(observation.quality.get("cohort", "unknown"))
            if feature in features:
                groups.setdefault(cohort, []).append(float(features[feature]))
        return {name: {"count": len(values), "mean": sum(values) / len(values)} for name, values in sorted(groups.items()) if values}

    def run_experiment(self, observations: list[Observation], *, feature: str) -> dict[str, Any]:
        """Run a fixed cohort-mean-spread summary instead of arbitrary code."""
        comparison = self.compare_cohorts(observations, feature=feature)
        means = [value["mean"] for value in comparison.values()]
        return {"experiment": "cohort_mean_spread", "feature": feature, "cohorts": comparison,
                "mean_spread": max(means) - min(means) if means else 0.0}

    def inspect_state(self, namespace: str | None = None, *, as_of: datetime | None = None) -> dict[str, Any]:
        """Read current state or a namespace snapshot at an event-time boundary."""
        self._query("inspect_state", {"namespace": namespace, "as_of": as_of})
        return self.state.snapshot_at(namespace, as_of) if namespace and as_of else self.state.snapshot(namespace)

    def propose_lens(self, candidate: LensCandidate) -> str:
        """Record a candidate definition and return its stable review ID."""
        self._query("propose_lens", {"candidate_id": candidate.candidate_id})
        self.state.put_candidate(candidate)
        return candidate.candidate_id

    def validate_lens(self, candidate_id: str, store: EventStore, *, validation_start: datetime, validation_end: datetime) -> dict[str, Any]:
        """Run bounded SQL and temporal/redundancy diagnostics on a candidate.

        The validator checks execution, output references, the declared unseen
        boundary, subwindow counts, cohort counts, and overlap with one active
        SQL lens. It cannot calculate false-positive rate without independent
        labels. A passing result is evidence for review, not approval or proof
        of semantic usefulness.
        """
        self._query("validate_lens", {"candidate_id": candidate_id, "validation_start": validation_start, "validation_end": validation_end})
        candidate = self.state.transition_candidate(candidate_id, CandidateStatus.TESTING.value)
        started = time.perf_counter()
        try:
            rows = store.query_sql_lens(candidate.sql, limit=self.query_limit, start=validation_start, end=validation_end)
            if validation_start < candidate.validation_boundary:
                raise ValueError("validation data precedes the candidate's declared unseen boundary")
            if any("event_id" not in row for row in rows):
                raise ValueError("candidate output must include event_id to retain evidence references")
            midpoint = validation_start + (validation_end - validation_start) / 2
            first_half = store.query_sql_lens(candidate.sql, limit=self.query_limit, start=validation_start, end=midpoint)
            second_half = store.query_sql_lens(candidate.sql, limit=self.query_limit, start=midpoint, end=validation_end)
            first_count, second_count = len(first_half), len(second_half)
            stability = abs(first_count - second_count) / max(first_count, second_count, 1)
            candidate_ids = {str(row["event_id"]) for row in rows}
            validation_events = store.native_events(validation_start, validation_end, self.query_limit)
            cohorts: dict[str, int] = {}
            for event in validation_events:
                if event.event_id in candidate_ids:
                    cohort = str(event.attributes.get("cohort", "unknown"))
                    cohorts[cohort] = cohorts.get(cohort, 0) + 1
            active_sql = [value for value in self.state.snapshot("lens").values()
                          if isinstance(value, dict) and value.get("status") == "active" and value.get("sql")]
            redundancy = None
            if active_sql:
                existing_rows = store.query_sql_lens(active_sql[0]["sql"], limit=self.query_limit,
                                                     start=validation_start, end=validation_end)
                existing_ids = {str(row["event_id"]) for row in existing_rows if "event_id" in row}
                union = candidate_ids | existing_ids
                redundancy = len(candidate_ids & existing_ids) / len(union) if union else 0.0
            checks = {"read_only_sql": True, "unseen_validation_boundary": True,
                      "event_references_present": True, "redundant_with_active_lens": bool(redundancy is not None and redundancy >= 0.98)}
            result = {"valid_sql": True, "validation_passed": not checks["redundant_with_active_lens"],
                      "output_count": len(rows), "validation_start": validation_start.isoformat(),
                      "validation_end": validation_end.isoformat(), "unseen_boundary": candidate.validation_boundary.isoformat(),
                      "cohort_output_counts": cohorts, "subwindow_output_counts": [first_count, second_count],
                      "subwindow_relative_difference": stability, "redundancy_jaccard": redundancy,
                      "false_positive_rate": None,
                      "false_positive_note": "Requires independent labels for the candidate's target behaviour.",
                      "checks": checks, "execution_cost_ms": round((time.perf_counter() - started) * 1000, 3)}
            if checks["redundant_with_active_lens"]:
                self.state.transition_candidate(candidate_id, CandidateStatus.REJECTED.value, validation=result)
                return result
        except Exception as error:
            result = {"valid_sql": False, "error": str(error), "execution_cost_ms": round((time.perf_counter() - started) * 1000, 3)}
            self.state.transition_candidate(candidate_id, CandidateStatus.REJECTED.value, validation=result)
            return result
        self.state.transition_candidate(candidate_id, CandidateStatus.VALIDATED.value, validation=result)
        return result

    def record_hypothesis(self, hypothesis: Hypothesis) -> str:
        """Persist a tentative explanation and return its ID."""
        self._query("record_hypothesis", {"hypothesis_id": hypothesis.hypothesis_id})
        self.state.put_hypothesis(hypothesis)
        return hypothesis.hypothesis_id


class LocalInvestigator:
    """No-model fallback that packages evidence and rival hypotheses for review.

    The fixed hypothesis text and confidence values are illustrative workflow
    outputs. They are not data-calibrated beliefs or validated explanations.
    """

    def __init__(self, tools: BoundedTools, state: StateStore, *,
                 feature_selector: Callable[[Observation], str | None] | None = None):
        """Bind bounded tools and an optional profile-specific feature choice.

        The core does not assume which feature is meaningful for a domain. A
        profile may inject a selector; when none is supplied, investigation
        still records evidence and hypotheses but skips the cohort comparison.
        """
        self.tools, self.state = tools, state
        self.feature_selector = feature_selector

    def investigate(self, observation: Observation, related: list[Observation] | None = None) -> dict[str, Any]:
        """Gather a bounded evidence packet and leave it for human review.

        The operation records supporting observations, nearby native events,
        a time-aware operational snapshot, an optional profile-selected cohort
        summary, and rival hypotheses. It never marks a hypothesis as verified.
        """
        started = time.perf_counter()
        self.tools.trace = []
        related = related or [observation]
        observed = self.tools.inspect_observations(related, limit=20)
        nearby = self.tools.query_events(start=observation.window_start, end=observation.window_end, limit=100)
        support = set(observation.supporting_event_ids)
        evidence_events = [event for event in nearby if event["event_id"] in support]
        operational_state = self.tools.inspect_state("operational", as_of=observation.window_end)
        features = observation.value.get("features", {}) if isinstance(observation.value, dict) else {}
        feature = self.feature_selector(observation) if self.feature_selector else None
        if feature and feature in features:
            experiment = self.tools.run_experiment(related, feature=feature)
        else:
            experiment = {
                "experiment": "cohort_mean_spread",
                "feature": feature,
                "cohorts": {},
                "mean_spread": 0.0,
                "skipped": "no profile selected an available numeric feature",
            }
        feature_hypothesis = (
            f"A change in {feature} is specific to a cohort."
            if feature else "A domain-relevant feature may have shifted for this subject."
        )
        candidates = [
            Hypothesis(statement=feature_hypothesis, supporting_evidence=(observation.observation_id,), confidence=0.35),
            Hypothesis(statement="The score reflects ordinary variation or a changed operating regime.",
                       contradicting_evidence=(observation.observation_id,), confidence=0.25),
        ]
        for hypothesis in candidates:
            self.tools.record_hypothesis(hypothesis)
        trace = {
            "investigation_id": str(uuid4()), "observation_id": observation.observation_id,
            "status": "deferred_for_human_review", "observed_count": len(observed),
            "feature_snapshot": features, "supporting_events": evidence_events,
            "operational_state": operational_state, "experiment": experiment,
            "hypotheses": [item.model_dump(mode="json") for item in candidates],
            "tool_trace": list(self.tools.trace), "inference_tokens": 0, "inference_cost": 0.0,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
            "note": "Local fallback generates structured evidence summaries; it does not validate causal explanations.",
        }
        with self.state.connection:
            self.state.connection.execute("INSERT INTO investigations VALUES (?,?)", (trace["investigation_id"], json.dumps(trace, default=str)))
        self.state.set("discovery", f"investigation:{trace['investigation_id']}", trace, "local bounded investigation")
        return trace
