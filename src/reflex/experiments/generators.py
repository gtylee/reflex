"""Seeded phase-one event generators with evaluator-only ground truth."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from reflex.core.models.events import Event, stable_id
from .models.inputs import ExperimentConfig, GroundTruthLabel


SCENARIOS: dict[str, dict[str, Any]] = {
    "exposure_drift": {"family": "1A", "feature": "exposure", "model_features": ("exposure",), "mechanism": "gradual_exposure_drift"},
    "collateral_deterioration": {"family": "1B", "feature": "collateral_composite", "model_features": ("coverage", "quality", "concentration", "liquidity"), "mechanism": "collateral_deterioration"},
    "rare_sequence": {"family": "1C", "feature": "sequence_score", "model_features": ("sequence_score",), "mechanism": "rare_multi_event_sequence"},
    "correlation_shift": {"family": "1D", "feature": "product_xy", "model_features": ("factor_x", "factor_y"), "mechanism": "correlation_regime_shift"},
    "spurious_confounder": {"family": "1E", "feature": "processing_load", "model_features": ("confounder", "processing_load"), "mechanism": "true_driver_after_confounder_break"},
    "unseen_mechanism": {"family": "1F", "feature": "partner_value_interaction", "model_features": ("amount", "partner_channel"), "mechanism": "novel_partner_value_relationship"},
}


@dataclass(frozen=True)
class ScenarioDataset:
    config: ExperimentConfig
    events: tuple[Event, ...]
    ground_truth: tuple[GroundTruthLabel, ...]
    start: datetime
    train_end: datetime
    calibration_end: datetime
    end: datetime
    input_hash: str

    def agent_input(self):
        from .models.inputs import AgentInput
        return AgentInput(
            case_id=f"case-{self.input_hash[:12]}", input_hash=self.input_hash,
            events=self.events, train_end=self.train_end,
            calibration_end=self.calibration_end, end=self.end, objective=self.config.objective,
        )

    def write(self, directory: str | Path, evaluator_dir: str | Path | None = None) -> dict[str, str]:
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        evaluator = Path(evaluator_dir) if evaluator_dir else root.parent / f"{root.name}-evaluator"
        evaluator.mkdir(parents=True, exist_ok=True)
        events_path = root / "events.jsonl"
        events_path.write_text("".join(json.dumps(event.model_dump(mode="json"), sort_keys=True) + "\n" for event in self.events))
        manifest_path = root / "experiment.json"
        manifest_path.write_text(json.dumps({
            "case_id": f"case-{self.input_hash[:12]}", "input_hash": self.input_hash,
            "event_count": len(self.events), "start": self.start.isoformat(),
            "train_end": self.train_end.isoformat(), "calibration_end": self.calibration_end.isoformat(),
            "end": self.end.isoformat(),
        }, indent=2) + "\n")
        truth_path = evaluator / "ground_truth.json"
        truth_path.write_text(json.dumps([item.model_dump(mode="json") for item in self.ground_truth], indent=2) + "\n")
        evaluator_config = evaluator / "experiment-config.json"
        evaluator_config.write_text(json.dumps(self.config.model_dump(mode="json"), indent=2) + "\n")
        return {"manifest": str(manifest_path), "events": str(events_path),
                "evaluator_truth": str(truth_path), "evaluator_config": str(evaluator_config),
                "input_hash": self.input_hash}


def default_configs(*, seed: int = 11, entities: int = 32, days: int = 30) -> tuple[ExperimentConfig, ...]:
    return tuple(ExperimentConfig(
        config_id=f"synthetic-{name}", version="1.0", phase=1, family=details["family"], scenario=name,
        objective="Identify event patterns requiring investigation, cite evidence, and avoid unsupported causal claims.",
        seed=seed, entity_count=entities, days=days,
        parameters={"detector_feature": details["feature"], "model_features": list(details["model_features"]),
                    "direction": "lower" if name == "correlation_shift" else "upper"},
    ) for name, details in SCENARIOS.items())


def generate_scenario(config: ExperimentConfig) -> ScenarioDataset:
    if config.scenario not in SCENARIOS:
        raise ValueError(f"unknown synthetic scenario {config.scenario!r}")
    rng = np.random.default_rng(config.seed)
    start = datetime(2025, 1, 1, tzinfo=timezone.utc)
    train_end = start + timedelta(days=config.days // 3)
    calibration_end = start + timedelta(days=(config.days * 2) // 3)
    end = start + timedelta(days=config.days)
    entities = [f"acct-{i:04d}" for i in range(config.entity_count)]
    target_entities = set(entities[:max(2, config.entity_count // 8)])
    shift_day = config.days // 2
    events: list[Event] = []
    truth: list[GroundTruthLabel] = []

    def emit(day: int, entity: str, sequence: int, event_type: str, attributes: dict[str, Any],
             *, minute: int = 0, mechanism: str | None = None, note: str = "") -> None:
        timestamp = start + timedelta(days=day, hours=10, minutes=minute, microseconds=sequence)
        event_id = stable_id("reflex-experiment", config.config_id, str(config.seed), str(day), entity,
                             str(sequence), event_type)
        events.append(Event(
            event_id=event_id, event_type=event_type, event_time=timestamp, entity_ids=(entity,),
            attributes=attributes, source="synthetic-experiment-v1",
            provenance={"seed": config.seed, "generator_version": "1.0"},
            relationships=(f"batch-{day // 5:02d}",),
        ))
        if mechanism:
            truth.append(GroundTruthLabel(event_id=event_id, entity_id=entity, mechanism=mechanism,
                                          event_time=timestamp, note=note))

    for day in range(config.days):
        for index, entity in enumerate(entities):
            if config.scenario == "exposure_drift":
                drift = (max(0, day - config.days // 4) * 0.018) if entity in target_entities else 0.0
                exposure = float(np.clip(0.22 + drift + rng.normal(0, 0.018), 0.05, 0.92))
                emit(day, entity, 0, "portfolio.exposure_mark", {
                    "counterparty": f"cp-{index % 7}", "exposure": round(exposure, 5), "static_limit": 0.95,
                    "daily_change": round(drift, 5), "region": f"region-{index % 4}",
                }, mechanism="gradual_exposure_drift" if entity in target_entities and day >= shift_day + 2 else None,
                    note="Exposure rises gradually while remaining below the conventional static limit.")
            elif config.scenario == "collateral_deterioration":
                affected = entity in target_entities
                progress = max(0, day - config.days // 4) if affected else 0
                coverage = float(1.18 - progress * 0.013 + rng.normal(0, 0.018))
                quality = float(0.92 - progress * 0.009 + rng.normal(0, 0.012))
                concentration = float(0.20 + progress * 0.013 + rng.normal(0, 0.015))
                liquidity = float(0.82 - progress * 0.010 + rng.normal(0, 0.015))
                stress = ((1.18 - coverage) / 0.25 + (0.92 - quality) / 0.20 +
                          (concentration - 0.20) / 0.30 + (0.82 - liquidity) / 0.25)
                emit(day, entity, 0, "collateral.valuation", {
                    "coverage": round(coverage, 5), "quality": round(quality, 5),
                    "concentration": round(concentration, 5), "liquidity": round(liquidity, 5),
                }, mechanism="collateral_deterioration" if affected and day >= shift_day + 2 and stress > 1.2 else None,
                    note="Several moderate collateral changes combine into elevated stress.")
            elif config.scenario == "rare_sequence":
                emit(day, entity, 0, "settlement.heartbeat", {"status": "normal", "sequence_score": 0.0})
                stages = ("settlement.file_received", "reconciliation.flagged",
                          "counterparty.confirmation_missing", "settlement.failed")
                # Each sequence component also appears on its own, so no single
                # event type or numeric field reveals the planted mechanism.
                benign_stage = stages[(day + index) % len(stages)]
                emit(day, entity, 10, benign_stage, {"sequence_score": 0.0,
                     "batch_ref": f"routine-{day}-{index}"}, minute=1)
                incident_days = {config.days // 3, (config.days * 2) // 3, config.days - 3}
                if entity in target_entities and day in incident_days:
                    for position, event_type in enumerate(stages, start=1):
                        emit(day, entity, position, event_type, {
                            "sequence_score": 0.0, "batch_ref": f"batch-{day}-{index}",
                            "amount": round(float(rng.lognormal(4.8, 0.2)), 2),
                        }, minute=position * 3,
                            mechanism="rare_multi_event_sequence" if position == len(stages) else None,
                            note="The event is only material after the ordered multi-event sequence.")
            elif config.scenario == "correlation_shift":
                x = float(rng.normal())
                rho = 0.82 if day < shift_day else -0.82
                y = float(rho * x + np.sqrt(1 - rho * rho) * rng.normal())
                emit(day, entity, 0, "market.factor_observation", {
                    "factor_x": round(x, 6), "factor_y": round(y, 6),
                    "product_xy": round(x * y, 6),
                }, mechanism="correlation_regime_shift"
                   if (config.days * 2) // 3 <= day < (config.days * 2) // 3 + 4 else None,
                    note="The marginal factor distributions remain stable as the correlation changes.")
            elif config.scenario == "spurious_confounder":
                confounder = int(rng.random() < (0.72 if day < shift_day else 0.50))
                if day < shift_day:
                    true_driver = int(rng.random() < 0.03)
                    outcome = int(true_driver or (confounder and rng.random() < 0.75) or rng.random() < 0.06)
                else:
                    true_driver = int(rng.random() < 0.16)
                    outcome = int(true_driver or rng.random() < 0.04)
                emit(day, entity, 0, "operational.outcome", {
                    "confounder": confounder, "outcome": outcome, "processing_load": int(true_driver),
                }, mechanism="true_driver_after_confounder_break" if true_driver and day >= shift_day else None,
                    note="After the correlation break, outcomes are generated by a different driver.")
            else:  # unseen_mechanism
                normal_amount = float(rng.lognormal(5.0, 0.20))
                new_relation = entity in target_entities and day >= shift_day
                channel = "partner_api" if new_relation else "web"
                emit(day, entity, 0, "portfolio.activity", {
                    "amount": round(normal_amount, 5), "channel": channel,
                    "counterparty": f"cp-{index % 9}",
                }, mechanism="novel_partner_value_relationship" if new_relation else None,
                    note="A new partner-channel/value relationship begins halfway through the stream.")

    events.sort(key=lambda event: (event.event_time, event.event_id))
    visible = json.dumps([event.model_dump(mode="json") for event in events], sort_keys=True, separators=(",", ":"))
    input_hash = sha256(visible.encode("utf-8")).hexdigest()
    return ScenarioDataset(config=config, events=tuple(events), ground_truth=tuple(truth), start=start,
                           train_end=train_end, calibration_end=calibration_end, end=end, input_hash=input_hash)


__all__ = ["SCENARIOS", "ScenarioDataset", "default_configs", "generate_scenario"]
