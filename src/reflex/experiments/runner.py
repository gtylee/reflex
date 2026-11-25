"""Fair shared-input comparisons, repeated runs, and evaluator-only scoring."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any, Iterable

import numpy as np

from .adapters import default_phase_one_adapters
from .generators import ScenarioDataset, generate_scenario
from .metrics import decision_variance, score_predictions
from .models.inputs import ExperimentConfig, RunBudget
from .models.results import (AdapterOutput, ExperimentAdapter, ExperimentReport, VARIANT_DESCRIPTIONS,
                             VariantRun)


def run_comparison(config: ExperimentConfig, dataset: ScenarioDataset,
                   adapters: dict[str, ExperimentAdapter] | None = None,
                   *, variants: Iterable[str] | None = None) -> ExperimentReport:
    if dataset.config.scenario != config.scenario or dataset.config.seed != config.seed:
        raise ValueError("experiment config and generated dataset do not match")
    selected = tuple(variants or config.variants)
    if len(selected) != len(set(selected)):
        raise ValueError("each variant may be run only once per comparison")
    unknown = set(selected) - set(VARIANT_DESCRIPTIONS)
    if unknown:
        raise ValueError(f"unknown experiment variant(s): {', '.join(sorted(unknown))}")
    registered = default_phase_one_adapters(config)
    registered.update(adapters or {})
    agent_input = dataset.agent_input()
    visible_ids = {event.event_id for event in agent_input.events}
    test_truth = tuple(label for label in dataset.ground_truth if label.event_time >= dataset.calibration_end)
    runs: list[VariantRun] = []

    for variant in selected:
        adapter = registered.get(variant)
        if adapter is None:
            runs.append(VariantRun(variant_id=variant, description=VARIANT_DESCRIPTIONS[variant],
                                  status="adapter_unavailable", input_hash=agent_input.input_hash))
            continue
        if getattr(adapter, "variant_id", variant) != variant:
            runs.append(VariantRun(variant_id=variant, description=VARIANT_DESCRIPTIONS[variant],
                                  status="failed", input_hash=agent_input.input_hash,
                                  error="adapter variant_id does not match the requested variant"))
            continue
        execution_kind = getattr(adapter, "execution_kind", "unclassified")
        if execution_kind not in {"deterministic", "external", "mock"}:
            execution_kind = "unclassified"
        try:
            wall_started = perf_counter()
            output = adapter.run(agent_input, config.budgets, trial_seed=config.seed)
            wall_latency = perf_counter() - wall_started
            invalid_predictions = set(output.predicted_event_ids) - visible_ids
            invalid_evidence = {ref for refs in output.evidence_refs.values() for ref in refs} - visible_ids
            if invalid_predictions or invalid_evidence:
                raise ValueError("adapter output refers to event IDs outside the shared input bundle")
            if (output.token_count > config.budgets.inference_tokens or
                    output.compute_seconds > config.budgets.compute_seconds or
                    output.tool_calls > config.budgets.tool_calls or
                    wall_latency > config.budgets.compute_seconds):
                runs.append(VariantRun(variant_id=variant, description=VARIANT_DESCRIPTIONS[variant],
                                      execution_kind=execution_kind, status="over_budget",
                                      input_hash=agent_input.input_hash, output=output,
                                      error="adapter-reported resource use exceeded the common budget"))
                continue
            metrics = score_predictions(output.predicted_event_ids, test_truth, output.evidence_refs,
                                        universe_event_ids=visible_ids,
                                        events_by_id={event.event_id: event for event in agent_input.events},
                                        bootstrap_seed=config.seed)
            metrics["inference_tokens"] = output.token_count
            metrics["compute_seconds"] = output.compute_seconds
            metrics["wall_latency_seconds"] = wall_latency
            metrics["cost_usd"] = output.cost_usd
            metrics["model_calls"] = output.model_calls
            metrics["tool_calls"] = output.tool_calls
            metrics["investigation_count"] = output.investigation_count
            metrics["time_to_first_discovery_seconds"] = _time_to_first_discovery(
                dataset, output.predicted_event_ids, test_truth)
            runs.append(VariantRun(variant_id=variant, description=VARIANT_DESCRIPTIONS[variant],
                                  execution_kind=execution_kind, status="completed", input_hash=agent_input.input_hash,
                                  metrics=metrics, output=output))
        except Exception as error:
            runs.append(VariantRun(variant_id=variant, description=VARIANT_DESCRIPTIONS[variant],
                                  execution_kind=execution_kind, status="failed",
                                  input_hash=agent_input.input_hash, error=str(error)))

    readiness_reasons = _readiness_reasons(runs)
    return ExperimentReport(
        config_id=config.config_id, config_version=config.version, scenario=config.scenario,
        seed=config.seed, input_hash=agent_input.input_hash, generated_event_count=len(dataset.events),
        truth_event_count=len(dataset.ground_truth), test_truth_event_count=len(test_truth), runs=tuple(runs),
        comparison_ready=(len(runs) >= 2 and not readiness_reasons),
        readiness_reasons=tuple(readiness_reasons),
    )


def run_config(config: ExperimentConfig, adapters: dict[str, ExperimentAdapter] | None = None,
               *, variants: Iterable[str] | None = None) -> tuple[ScenarioDataset, ExperimentReport]:
    dataset = generate_scenario(config)
    return dataset, run_comparison(config, dataset, adapters, variants=variants)


def write_report(report: ExperimentReport, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report.model_dump(mode="json"), indent=2) + "\n")
    return target


def repeat_adapter(adapter: ExperimentAdapter, dataset: ScenarioDataset, budgets: RunBudget,
                   *, repeats: int = 20, first_trial_seed: int = 1000) -> dict[str, Any]:
    """Run the same event bundle repeatedly; never substitutes mock runs for a real adapter."""
    if repeats < 2:
        raise ValueError("at least two runs are required to estimate variance")
    agent_input = dataset.agent_input()
    test_truth = tuple(label for label in dataset.ground_truth if label.event_time >= dataset.calibration_end)
    outcomes: list[str] = []
    costs: list[float] = []
    token_counts: list[int] = []
    f1_scores: list[float] = []
    trial_metrics: list[dict[str, Any]] = []
    runs = []
    latencies = []
    for index in range(repeats):
        wall_started = perf_counter()
        output = adapter.run(agent_input, budgets, trial_seed=first_trial_seed + index)
        wall_latency = perf_counter() - wall_started
        latencies.append(wall_latency)
        if (output.token_count > budgets.inference_tokens or output.compute_seconds > budgets.compute_seconds or
                output.tool_calls > budgets.tool_calls or wall_latency > budgets.compute_seconds):
            raise ValueError("repeated adapter run exceeded the shared resource budget")
        visible_ids = {event.event_id for event in agent_input.events}
        invalid_predictions = set(output.predicted_event_ids) - visible_ids
        invalid_evidence = {ref for refs in output.evidence_refs.values() for ref in refs} - visible_ids
        if invalid_predictions or invalid_evidence:
            raise ValueError("adapter returned event/evidence IDs outside the common input bundle")
        outcome = json.dumps({"predictions": sorted(output.predicted_event_ids),
                              "actions": output.actions, "discoveries": output.discoveries},
                             sort_keys=True, default=str)
        outcomes.append(outcome)
        costs.append(output.compute_seconds)
        token_counts.append(output.token_count)
        metrics = score_predictions(output.predicted_event_ids, test_truth, output.evidence_refs,
                                    universe_event_ids=(event.event_id for event in agent_input.events),
                                    events_by_id={event.event_id: event for event in agent_input.events},
                                    bootstrap_seed=first_trial_seed + index)
        f1_scores.append(float(metrics["f1"]))
        trial_metrics.append(metrics)
        runs.append(output)
    return {
        "adapter": getattr(adapter, "variant_id", adapter.__class__.__name__),
        "execution_kind": getattr(adapter, "execution_kind", "unclassified"),
        "input_hash": agent_input.input_hash, "runs": repeats,
        "business_outcome_variance": decision_variance(outcomes),
        "numerical_output_variance": decision_variance([item.output_variance_value for item in runs
                                                          if item.output_variance_value is not None]),
        "correctness": {"mean_f1": sum(f1_scores) / len(f1_scores),
                        "f1_variance": float(np.var(f1_scores)),
                        "trials": trial_metrics},
        "mean_compute_seconds": sum(costs) / len(costs), "mean_inference_tokens": sum(token_counts) / len(token_counts),
        "mean_wall_latency_seconds": sum(latencies) / len(latencies),
        "mean_cost_usd": sum(output.cost_usd for output in runs) / len(runs),
        "outputs": [item.model_dump(mode="json") for item in runs],
        "note": "Reasoning-path variation is not measured here; only structured business outputs are compared.",
    }


def _time_to_first_discovery(dataset: ScenarioDataset, predicted_event_ids: Iterable[str], truth) -> float | None:
    truth_by_id = {label.event_id: label.event_time for label in truth}
    matched = [truth_by_id[event_id] for event_id in predicted_event_ids if event_id in truth_by_id]
    if not matched or not truth_by_id:
        return None
    first_truth = min(truth_by_id.values())
    return max(0.0, (min(matched) - first_truth).total_seconds())


def _readiness_reasons(runs: list[VariantRun]) -> list[str]:
    reasons = []
    if len(runs) < 2:
        reasons.append("at least two variants are required")
    incomplete = [run.variant_id for run in runs
                  if run.status != "completed" or run.execution_kind not in {"deterministic", "external"}]
    if incomplete:
        reasons.append("not completed with real/deterministic execution: " + ", ".join(incomplete))
    external = [run for run in runs if run.execution_kind == "external" and run.output]
    for run in external:
        metadata = run.output.metadata
        missing = {"model_version", "sampling_parameters"} - set(metadata)
        if run.variant_id in {"B", "G"}:
            missing |= {"tool_versions"} - set(metadata)
        if run.variant_id == "G":
            missing |= {"runtime_version"} - set(metadata)
        if missing:
            reasons.append(f"{run.variant_id} external provenance is missing: {', '.join(sorted(missing))}")
    model_versions = {run.output.metadata.get("model_version") for run in external
                      if run.variant_id in {"A", "B"}}
    if len(model_versions) > 1:
        reasons.append("A/B model versions do not match")
    sampling = {json.dumps(run.output.metadata.get("sampling_parameters"), sort_keys=True)
                for run in external if run.variant_id in {"A", "B"}}
    if len(sampling) > 1:
        reasons.append("A/B sampling parameters do not match")
    return reasons


__all__ = ["run_comparison", "run_config", "write_report", "repeat_adapter"]
