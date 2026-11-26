"""Runnable deterministic analysis adapters for the shared experiment inputs."""

from __future__ import annotations

from datetime import timedelta
from time import perf_counter
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
import torch
from torch import nn

from .models.inputs import AgentInput, RunBudget
from .models.results import AdapterOutput


def feature_value(event, feature: str) -> float:
    attrs = event.attributes
    if feature == "collateral_composite":
        return float((1.18 - float(attrs["coverage"])) / 0.25 +
                     (0.92 - float(attrs["quality"])) / 0.20 +
                     (float(attrs["concentration"]) - 0.20) / 0.30 +
                     (0.82 - float(attrs["liquidity"])) / 0.25)
    if feature == "partner_value_interaction":
        return float(attrs["amount"]) / 150.0 if attrs["channel"] == "partner_api" else 0.0
    if feature == "partner_channel":
        return 1.0 if attrs.get("channel") == "partner_api" else 0.0
    if feature == "failure_sequence":
        return 1.0 if event.event_type == "settlement.failed" else 0.0
    return float(attrs.get(feature, 0.0))


class QuantileLensAdapter:
    """Transparent univariate lens. Fits only on the declared training interval."""

    variant_id = "C"
    execution_kind = "deterministic"

    def __init__(self, feature: str, *, direction: str = "upper", quantile: float = 0.995):
        if direction not in {"upper", "lower"} or not 0.5 < quantile < 1.0:
            raise ValueError("direction must be upper/lower and quantile must be between .5 and 1")
        self.feature = feature
        self.direction = direction
        self.quantile = quantile

    def run(self, experiment: AgentInput, budgets: RunBudget, *, trial_seed: int) -> AdapterOutput:
        started = perf_counter()
        train = [event for event in experiment.events if event.event_time < experiment.train_end]
        test = [event for event in experiment.events if event.event_time >= experiment.calibration_end]
        values = np.asarray([feature_value(event, self.feature) for event in train], dtype=float)
        if not len(values):
            return AdapterOutput(compute_seconds=perf_counter() - started,
                                 metadata={"detector": "quantile", "error": "no training events"})
        percentile = self.quantile if self.direction == "upper" else 1 - self.quantile
        threshold = float(np.quantile(values, percentile))
        if self.direction == "upper":
            inclusive = threshold >= float(values.max()) and threshold > 0
            predicted = [event for event in test
                         if (feature_value(event, self.feature) >= threshold if inclusive
                             else feature_value(event, self.feature) > threshold)]
        else:
            inclusive = threshold <= float(values.min()) and threshold < 0
            predicted = [event for event in test
                         if (feature_value(event, self.feature) <= threshold if inclusive
                             else feature_value(event, self.feature) < threshold)]
        ids = tuple(event.event_id for event in predicted)
        return AdapterOutput(predicted_event_ids=ids,
                             evidence_refs={event.event_id: (event.event_id,) for event in predicted},
                             compute_seconds=perf_counter() - started,
                             metadata={"detector": "quantile", "feature": self.feature,
                                       "direction": self.direction, "threshold": threshold,
                                       "train_event_count": len(train), "test_event_count": len(test),
                                       "trial_seed": trial_seed})


class IsolationForestAdapter:
    """Multivariate detector over declared numeric attributes."""

    variant_id = "C"
    execution_kind = "deterministic"

    def __init__(self, features: tuple[str, ...], *, contamination: float = 0.02, random_state: int = 11):
        if not features or not 0 < contamination < 0.5:
            raise ValueError("features are required and contamination must be in (0, .5)")
        self.features, self.contamination, self.random_state = features, contamination, random_state

    def run(self, experiment: AgentInput, budgets: RunBudget, *, trial_seed: int) -> AdapterOutput:
        started = perf_counter()
        train = [event for event in experiment.events if event.event_time < experiment.train_end]
        test = [event for event in experiment.events if event.event_time >= experiment.calibration_end]
        if len(train) < 2 or not test:
            return AdapterOutput(compute_seconds=perf_counter() - started,
                                 metadata={"detector": "isolation_forest", "error": "insufficient data"})
        x_train = np.asarray([[feature_value(event, name) for name in self.features] for event in train], dtype=float)
        x_test = np.asarray([[feature_value(event, name) for name in self.features] for event in test], dtype=float)
        model = IsolationForest(contamination=self.contamination, random_state=self.random_state + trial_seed)
        model.fit(x_train)
        mask = model.predict(x_test) < 0
        predicted = [event for event, flagged in zip(test, mask, strict=True) if flagged]
        return AdapterOutput(predicted_event_ids=tuple(event.event_id for event in predicted),
                             evidence_refs={event.event_id: (event.event_id,) for event in predicted},
                             compute_seconds=perf_counter() - started,
                             metadata={"detector": "isolation_forest", "features": list(self.features),
                                       "train_event_count": len(train), "test_event_count": len(test),
                                       "random_state": self.random_state + trial_seed})


class GaussianMixtureAdapter:
    """Training-only Gaussian mixture score with a held-out training quantile."""

    variant_id = "C"
    execution_kind = "deterministic"

    def __init__(self, features: tuple[str, ...], *, components: int = 2, contamination: float = 0.02):
        if not features or components < 1 or not 0 < contamination < 0.5:
            raise ValueError("invalid Gaussian mixture settings")
        self.features, self.components, self.contamination = features, components, contamination

    def run(self, experiment: AgentInput, budgets: RunBudget, *, trial_seed: int) -> AdapterOutput:
        started = perf_counter()
        train = [event for event in experiment.events if event.event_time < experiment.train_end]
        test = [event for event in experiment.events if event.event_time >= experiment.calibration_end]
        x_train = np.asarray([[feature_value(event, name) for name in self.features] for event in train], dtype=float)
        x_test = np.asarray([[feature_value(event, name) for name in self.features] for event in test], dtype=float)
        if len(train) <= self.components or not len(test):
            return AdapterOutput(compute_seconds=perf_counter() - started,
                                 metadata={"detector": "gaussian_mixture", "error": "insufficient data"})
        model = GaussianMixture(n_components=self.components, random_state=trial_seed, reg_covar=1e-4)
        model.fit(x_train)
        threshold = float(np.quantile(model.score_samples(x_train), self.contamination))
        predicted = [event for event, score in zip(test, model.score_samples(x_test), strict=True) if score < threshold]
        return AdapterOutput(predicted_event_ids=tuple(event.event_id for event in predicted),
                             evidence_refs={event.event_id: (event.event_id,) for event in predicted},
                             compute_seconds=perf_counter() - started,
                             metadata={"detector": "gaussian_mixture", "features": list(self.features),
                                       "threshold": threshold, "trial_seed": trial_seed})


class AutoencoderAdapter:
    """Small CPU neural reconstruction lens; all fitting uses training events only."""
    variant_id = "C"
    execution_kind = "deterministic"

    def __init__(self, features: tuple[str, ...], *, contamination: float = 0.02, epochs: int = 50):
        if not features or not 0 < contamination < 0.5 or epochs < 1:
            raise ValueError("invalid autoencoder settings")
        self.features, self.contamination, self.epochs = features, contamination, epochs

    def run(self, experiment: AgentInput, budgets: RunBudget, *, trial_seed: int) -> AdapterOutput:
        started = perf_counter()
        train = [event for event in experiment.events if event.event_time < experiment.train_end]
        test = [event for event in experiment.events if event.event_time >= experiment.calibration_end]
        if len(train) < 4 or not test:
            return AdapterOutput(compute_seconds=perf_counter() - started,
                                 metadata={"detector": "autoencoder", "error": "insufficient data"})
        x_train = np.asarray([[feature_value(event, key) for key in self.features] for event in train], dtype=np.float32)
        x_test = np.asarray([[feature_value(event, key) for key in self.features] for event in test], dtype=np.float32)
        mean, scale = x_train.mean(axis=0), x_train.std(axis=0)
        scale[scale < 1e-6] = 1.0
        x_train = (x_train - mean) / scale
        x_test = (x_test - mean) / scale
        torch.manual_seed(trial_seed)
        torch.set_num_threads(1)
        dims = x_train.shape[1]
        hidden = max(1, min(16, dims * 2))
        model = nn.Sequential(nn.Linear(dims, hidden), nn.Tanh(), nn.Linear(hidden, dims))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
        training_tensor = torch.as_tensor(x_train)
        for _ in range(self.epochs):
            reconstruction = model(training_tensor)
            loss = torch.mean((reconstruction - training_tensor) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            train_error = ((model(training_tensor) - training_tensor) ** 2).mean(dim=1).numpy()
            test_tensor = torch.as_tensor(x_test)
            test_error = ((model(test_tensor) - test_tensor) ** 2).mean(dim=1).numpy()
        threshold = float(np.quantile(train_error, 1 - self.contamination))
        predicted = [event for event, error in zip(test, test_error, strict=True) if error > threshold]
        return AdapterOutput(predicted_event_ids=tuple(event.event_id for event in predicted),
                             evidence_refs={event.event_id: (event.event_id,) for event in predicted},
                             compute_seconds=perf_counter() - started,
                             metadata={"detector": "autoencoder", "features": list(self.features),
                                       "threshold": threshold, "epochs": self.epochs, "trial_seed": trial_seed})


class CorrelationShiftAdapter:
    """Detect rolling correlation deviations from the training correlation."""
    variant_id = "C"
    execution_kind = "deterministic"

    def __init__(self, left: str = "factor_x", right: str = "factor_y", *, window: int = 16,
                 minimum_change: float = 0.35):
        if window < 4 or not 0 < minimum_change < 2:
            raise ValueError("invalid rolling correlation settings")
        self.left, self.right, self.window, self.minimum_change = left, right, window, minimum_change

    def run(self, experiment: AgentInput, budgets: RunBudget, *, trial_seed: int) -> AdapterOutput:
        started = perf_counter()
        events = sorted(experiment.events, key=lambda event: (event.event_time, event.event_id))
        train = [event for event in events if event.event_time < experiment.train_end]
        if len(train) < self.window:
            return AdapterOutput(compute_seconds=perf_counter() - started,
                                 metadata={"detector": "rolling_correlation", "error": "insufficient training window"})
        base = float(np.corrcoef([feature_value(e, self.left) for e in train],
                                 [feature_value(e, self.right) for e in train])[0, 1])
        later = [event for event in events if event.event_time >= experiment.calibration_end]
        evidence: dict[str, tuple[str, ...]] = {}
        for index in range(self.window, len(later) + 1):
            sample = later[index - self.window:index]
            corr = float(np.corrcoef([feature_value(e, self.left) for e in sample],
                                     [feature_value(e, self.right) for e in sample])[0, 1])
            if np.isfinite(corr) and abs(corr - base) >= self.minimum_change:
                sample_ids = tuple(event.event_id for event in sample)
                for event in sample:
                    evidence[event.event_id] = sample_ids
        selected = [event for event in later if event.event_id in evidence]
        return AdapterOutput(predicted_event_ids=tuple(event.event_id for event in selected),
                             evidence_refs={event.event_id: evidence[event.event_id] for event in selected},
                             compute_seconds=perf_counter() - started,
                             metadata={"detector": "rolling_correlation", "baseline_correlation": base,
                                       "window": self.window, "minimum_change": self.minimum_change})


class CalibratedCorrelationShiftAdapter:
    """Fit a baseline correlation, calibrate a rolling-change cutoff, then score held-out events.

    The calibration interval is required and remains label-blind. Evidence
    references point to the rolling cross-entity cohort window, not a single
    entity-local cause. A valid run needs enough observations for a calibration
    window.
    """

    variant_id = "C"
    execution_kind = "deterministic"

    def __init__(self, left: str = "factor_x", right: str = "factor_y", *,
                 window: int = 24, quantile: float = 0.99):
        if window < 4 or not 0.5 < quantile < 1.0:
            raise ValueError("window must be at least four and quantile must be between .5 and 1")
        self.left, self.right, self.window, self.quantile = left, right, window, quantile

    def run(self, experiment: AgentInput, budgets: RunBudget, *, trial_seed: int) -> AdapterOutput:
        started = perf_counter()
        events = sorted(experiment.events, key=lambda event: (event.event_time, event.event_id))
        train = [event for event in events if event.event_time < experiment.train_end]
        calibration = [event for event in events
                       if experiment.train_end <= event.event_time < experiment.calibration_end]
        if len(train) < self.window or len(calibration) < self.window:
            return AdapterOutput(compute_seconds=perf_counter() - started,
                                 metadata={"detector": "calibrated_rolling_correlation",
                                           "error": "insufficient training or calibration window"})
        base = float(np.corrcoef([feature_value(e, self.left) for e in train],
                                 [feature_value(e, self.right) for e in train])[0, 1])
        scores: dict[str, float] = {}
        windows: dict[str, tuple[str, ...]] = {}
        for index in range(self.window - 1, len(events)):
            sample = events[index - self.window + 1:index + 1]
            corr = float(np.corrcoef([feature_value(e, self.left) for e in sample],
                                     [feature_value(e, self.right) for e in sample])[0, 1])
            if np.isfinite(corr):
                target = events[index]
                scores[target.event_id] = abs(corr - base)
                windows[target.event_id] = tuple(event.event_id for event in sample)
        cal_scores = [scores[event.event_id] for event in calibration if event.event_id in scores]
        if not cal_scores:
            return AdapterOutput(compute_seconds=perf_counter() - started,
                                 metadata={"detector": "calibrated_rolling_correlation",
                                           "error": "no valid calibration scores"})
        threshold = float(np.quantile(cal_scores, self.quantile))
        test = [event for event in events if event.event_time >= experiment.calibration_end]
        predicted = [event for event in test
                     if scores.get(event.event_id, float("-inf")) > threshold]
        return AdapterOutput(predicted_event_ids=tuple(event.event_id for event in predicted),
                             evidence_refs={event.event_id: windows[event.event_id] for event in predicted},
                             compute_seconds=perf_counter() - started,
                             metadata={"detector": "calibrated_rolling_correlation",
                                       "baseline_correlation": base, "threshold": threshold,
                                       "threshold_quantile": self.quantile,
                                       "window_events": self.window,
                                       "evidence_scope": "cross_entity_factor_cohort",
                                       "training_event_count": len(train),
                                       "calibration_event_count": len(calibration),
                                       "test_event_count": len(test), "trial_seed": trial_seed})


class SequencePatternAdapter:
    """Detect an ordered event pattern and flag only its completing event."""
    variant_id = "C"
    execution_kind = "deterministic"

    def __init__(self, pattern: tuple[str, ...], *, max_gap: timedelta = timedelta(hours=1)):
        if len(pattern) < 3:
            raise ValueError("sequence patterns must contain at least three event types")
        if max_gap <= timedelta(0):
            raise ValueError("maximum sequence gap must be positive")
        self.pattern = pattern
        self.max_gap = max_gap

    def run(self, experiment: AgentInput, budgets: RunBudget, *, trial_seed: int) -> AdapterOutput:
        started = perf_counter()
        by_entity: dict[str, list[Any]] = {}
        for event in sorted(experiment.events, key=lambda item: (item.event_time, item.event_id)):
            if event.event_time >= experiment.calibration_end:
                for entity in event.entity_ids:
                    by_entity.setdefault(entity, []).append(event)
        predicted, evidence = [], {}
        for values in by_entity.values():
            matched: list[Any] = []
            next_index = 0
            for event in values:
                if matched and event.event_time - matched[-1].event_time > self.max_gap:
                    matched, next_index = [], 0
                if event.event_type == self.pattern[next_index]:
                    matched.append(event)
                    next_index += 1
                    if next_index == len(self.pattern):
                        predicted.append(event)
                        evidence[event.event_id] = tuple(item.event_id for item in matched)
                        matched, next_index = [], 0
                elif event.event_type == self.pattern[0]:
                    matched, next_index = [event], 1
                else:
                    matched, next_index = [], 0
        return AdapterOutput(predicted_event_ids=tuple(event.event_id for event in predicted),
                             evidence_refs=evidence, compute_seconds=perf_counter() - started,
                             metadata={"detector": "sequence_pattern", "pattern": list(self.pattern),
                                       "max_gap_seconds": self.max_gap.total_seconds()})


class CompositeLensAdapter:
    """Combine versioned deterministic lenses under one divided resource budget."""
    variant_id = "C"
    execution_kind = "deterministic"

    def __init__(self, adapters: tuple[Any, ...], *, strategy: str = "any"):
        if not adapters or strategy not in {"any", "all"}:
            raise ValueError("provide lenses and choose strategy 'any' or 'all'")
        if any(getattr(adapter, "execution_kind", None) != "deterministic" for adapter in adapters):
            raise ValueError("composite lenses must be deterministic adapters")
        self.adapters, self.strategy = adapters, strategy

    def run(self, experiment: AgentInput, budgets: RunBudget, *, trial_seed: int) -> AdapterOutput:
        child_budgets = RunBudget(inference_tokens=budgets.inference_tokens // len(self.adapters),
                                  compute_seconds=budgets.compute_seconds / len(self.adapters),
                                  tool_calls=budgets.tool_calls // len(self.adapters))
        outputs = [adapter.run(experiment, child_budgets, trial_seed=trial_seed)
                   for adapter in self.adapters]
        id_sets = [set(output.predicted_event_ids) for output in outputs]
        selected = set.union(*id_sets) if self.strategy == "any" else set.intersection(*id_sets)
        evidence = {event_id: tuple(dict.fromkeys(ref for output in outputs
                                                  for ref in output.evidence_refs.get(event_id, ())))
                    for event_id in selected}
        return AdapterOutput(predicted_event_ids=tuple(sorted(selected)), evidence_refs=evidence,
                             token_count=sum(output.token_count for output in outputs),
                             compute_seconds=sum(output.compute_seconds for output in outputs),
                             model_calls=sum(output.model_calls for output in outputs),
                             tool_calls=sum(output.tool_calls for output in outputs),
                             investigation_count=sum(output.investigation_count for output in outputs),
                             discoveries=tuple(item for output in outputs for item in output.discoveries),
                             actions=tuple(item for output in outputs for item in output.actions),
                             metadata={"detector": "composite", "strategy": self.strategy,
                                       "children": [output.metadata.get("detector", adapter.__class__.__name__)
                                                    for adapter, output in zip(self.adapters, outputs, strict=True)]})


ANALYTICAL_TOOLBOX = ("threshold", "isolation_forest", "gaussian_mixture", "autoencoder",
                      "rolling_correlation_change_point", "event_sequence", "clustering")


def default_phase_one_adapters(config) -> dict[str, Any]:
    if config.scenario == "rare_sequence":
        return {"C": SequencePatternAdapter(("settlement.file_received", "reconciliation.flagged",
                                               "counterparty.confirmation_missing", "settlement.failed"))}
    if (config.scenario == "correlation_shift"
            and config.parameters.get("lens_strategy") == "calibrated_rolling_correlation"):
        return {"C": CalibratedCorrelationShiftAdapter()}
    feature = str(config.parameters["detector_feature"])
    direction = str(config.parameters.get("direction", "upper"))
    return {"C": QuantileLensAdapter(feature, direction=direction)}


__all__ = ["feature_value", "QuantileLensAdapter", "IsolationForestAdapter", "GaussianMixtureAdapter",
           "AutoencoderAdapter", "CorrelationShiftAdapter", "CalibratedCorrelationShiftAdapter",
           "SequencePatternAdapter", "CompositeLensAdapter", "ANALYTICAL_TOOLBOX",
           "default_phase_one_adapters"]
