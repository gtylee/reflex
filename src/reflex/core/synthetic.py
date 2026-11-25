"""Lazy compatibility bridge for the former synthetic core module.

Synthetic data generation and hidden evaluator labels are experiment concerns.
New code should import these types from ``reflex.experiments.synthetic_data``.
This module remains as a small bridge for existing local scripts.
"""

from importlib import import_module

__all__ = ["SyntheticDataset", "generate_synthetic", "write_dataset"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module("reflex.experiments.synthetic_data")
    value = getattr(module, name)
    globals()[name] = value
    return value
