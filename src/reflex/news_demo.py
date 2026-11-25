"""Compatibility imports for the synthetic portfolio-news experiment.

New code should import the demo from ``reflex.experiments.news_demo``.
Reusable Reaction contracts and runtime behavior live outside the experiment.
"""

from reflex.experiments.news_demo import (
    benchmark_input_hash,
    benchmark_inputs,
    build_portfolio_news_reaction,
    run_news_demo,
)

__all__ = [
    "benchmark_input_hash",
    "benchmark_inputs",
    "build_portfolio_news_reaction",
    "run_news_demo",
]
