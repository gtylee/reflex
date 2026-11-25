"""Lazy compatibility bridge for the former experiment-scoring core module.

Synthetic evaluator labels and A–E benchmark records live in
``reflex.experiments.evaluation``. New code should import them from there.
"""

from importlib import import_module

__all__ = [
    "VARIANTS", "append_run_record", "mean_interval", "protocol_manifest",
    "protocol_report", "score_synthetic_run",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module("reflex.experiments.evaluation")
    value = getattr(module, name)
    globals()[name] = value
    return value
