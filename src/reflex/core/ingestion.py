"""Lazy compatibility bridge for the former dataset-specific core adapters.

New code should import BPI and Loghub readers from
``reflex.experiments.log_adapters``. Importing ``reflex.core`` does not load
these adapters; legacy callers that request one still receive it on demand.
"""

from importlib import import_module

__all__ = ["read_bpi2017", "read_loghub"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module("reflex.experiments.log_adapters")
    value = getattr(module, name)
    globals()[name] = value
    return value
