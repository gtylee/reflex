"""Compatibility imports for the former experiment-scoring core module.

Synthetic evaluator labels and A–E benchmark records live in
``reflex.experiments.evaluation``. New code should import them from there.
"""

from reflex.experiments.evaluation import (
    VARIANTS,
    append_run_record,
    mean_interval,
    protocol_manifest,
    protocol_report,
    score_synthetic_run,
)

__all__ = [
    "VARIANTS", "append_run_record", "mean_interval", "protocol_manifest",
    "protocol_report", "score_synthetic_run",
]
