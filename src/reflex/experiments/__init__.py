"""Reproducible REFLEX experiments; labels are kept outside baseline inputs."""

from .adapters import (ANALYTICAL_TOOLBOX, AutoencoderAdapter, CompositeLensAdapter, CorrelationShiftAdapter,
                       GaussianMixtureAdapter, IsolationForestAdapter, QuantileLensAdapter,
                       SequencePatternAdapter)
from .generators import SCENARIOS, ScenarioDataset, default_configs, generate_scenario
from .metrics import decision_variance, mean_interval, score_predictions
from .models import AgentInput, AdapterOutput, ExperimentConfig, ExperimentReport, GroundTruthLabel, RunBudget
from .runner import repeat_adapter, run_comparison, run_config, write_report
from .heldout import run_heldout_benchmark, write_heldout_report
from .lens_selection import LensSelection, select_lens, simple_numeric_baseline
from .datasets import (adapt_fnspid_rows, adapt_gdelt_rows, adapt_market_price_rows,
                       download_gdelt_day, parse_reuters21578)
from .harnesses import ElizaExperimentAdapter, ElizaHarness, MockElizaHarness
from .performance import benchmark_event_store
from .public_replay import replay_gdelt_day
from .visualizations import write_metric_svg, write_news_timeline_svg
from .workflows import (OperationalDataset, PortfolioExposure, RepairSandbox,
                        SimulatedOperationsSystem, diagnose_incident, apply_feedback,
                        evaluate_topic_association, process_news_events, process_news_timeline, synthetic_feedback,
                        synthetic_incidents, synthetic_news_outcomes, synthetic_news_story,
                        validate_candidate)

__all__ = [
    "SCENARIOS", "ScenarioDataset", "default_configs", "generate_scenario", "GroundTruthLabel",
    "AgentInput", "AdapterOutput", "ExperimentConfig", "ExperimentReport", "RunBudget",
    "QuantileLensAdapter", "IsolationForestAdapter", "GaussianMixtureAdapter", "AutoencoderAdapter",
    "CorrelationShiftAdapter", "SequencePatternAdapter", "CompositeLensAdapter", "ANALYTICAL_TOOLBOX", "run_comparison",
    "run_config", "write_report", "repeat_adapter", "score_predictions", "mean_interval", "decision_variance",
    "run_heldout_benchmark", "write_heldout_report",
    "LensSelection", "select_lens", "simple_numeric_baseline",
    "adapt_gdelt_rows", "adapt_fnspid_rows", "adapt_market_price_rows", "parse_reuters21578",
    "download_gdelt_day", "replay_gdelt_day",
    "ElizaHarness", "MockElizaHarness", "ElizaExperimentAdapter", "benchmark_event_store", "PortfolioExposure",
    "OperationalDataset", "synthetic_incidents", "diagnose_incident", "RepairSandbox", "SimulatedOperationsSystem",
    "synthetic_news_story", "process_news_events", "process_news_timeline", "synthetic_news_outcomes",
    "evaluate_topic_association", "synthetic_feedback", "validate_candidate",
    "apply_feedback",
    "write_metric_svg", "write_news_timeline_svg",
]
