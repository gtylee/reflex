"""Optional REFLEX experiment profiles and evaluation helpers.

This package facade keeps importing a specific profile lightweight. Public
names remain available from ``reflex.experiments`` and load their owning
module only when requested.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_MODULE_EXPORTS = {
    "adapters": (
        "ANALYTICAL_TOOLBOX", "AutoencoderAdapter", "CompositeLensAdapter",
        "CorrelationShiftAdapter", "GaussianMixtureAdapter", "IsolationForestAdapter",
        "QuantileLensAdapter", "SequencePatternAdapter",
    ),
    "generators": ("SCENARIOS", "ScenarioDataset", "default_configs", "generate_scenario"),
    "metrics": ("decision_variance", "mean_interval", "score_predictions"),
    "models": ("AgentInput", "AdapterOutput", "ExperimentConfig", "ExperimentReport", "GroundTruthLabel", "RunBudget"),
    "runner": ("repeat_adapter", "run_comparison", "run_config", "write_report"),
    "heldout": ("run_heldout_benchmark", "write_heldout_report"),
    "lens_selection": ("LensSelection", "select_lens", "simple_numeric_baseline"),
    "datasets": (
        "adapt_fnspid_rows", "adapt_gdelt_rows", "adapt_market_price_rows",
        "download_gdelt_day", "parse_reuters21578",
    ),
    "harnesses": ("ElizaExperimentAdapter", "ElizaHarness", "MockElizaHarness"),
    "performance": ("benchmark_event_store",),
    "public_replay": ("replay_gdelt_day",),
    "visualizations": ("write_metric_svg", "write_news_timeline_svg"),
    "workflows": (
        "OperationalDataset", "PortfolioExposure", "RepairSandbox", "SimulatedOperationsSystem",
        "diagnose_incident", "apply_feedback", "evaluate_topic_association", "process_news_events",
        "process_news_timeline", "synthetic_feedback", "synthetic_incidents",
        "synthetic_news_outcomes", "synthetic_news_story", "validate_candidate",
    ),
}

_EXPORTS = {name: module for module, names in _MODULE_EXPORTS.items() for name in names}
__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    """Load a public experiment helper from the module that owns it."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazy public helpers in interactive package discovery."""
    return sorted(set(globals()) | set(__all__))
