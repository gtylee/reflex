"""Generic numeric conversion for feature records.

Feature engineering belongs to a source adapter or experiment profile. This
module deliberately knows only that an ``EventWindowFeatures`` record contains
a mapping of feature names to numeric values.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .models.features import EventWindowFeatures


def matrix(rows: list[EventWindowFeatures], names: tuple[str, ...] | None = None) -> np.ndarray:
    """Convert typed feature rows to a dense matrix in an explicit schema order.

    When ``names`` is omitted, the first row's insertion-ordered feature keys
    define the schema. Fitting code should save that order with its model and
    pass it back while scoring later rows.
    """
    if names is None:
        names = tuple(rows[0].feature_values) if rows else ()
    return np.asarray([[row.feature_values[name] for name in names] for row in rows], dtype=np.float64)


def __getattr__(name: str):
    """Lazily expose old transaction helpers while callers migrate imports.

    The compatibility path keeps existing examples running without making the
    reusable feature matrix import transaction-specific experiment code.
    New code should import these names from
    ``reflex.experiments.transaction_features``.
    """
    legacy_names = {"FEATURE_NAMES", "event_level_features", "build_entity_windows"}
    if name not in legacy_names:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    transaction_features = import_module("reflex.experiments.transaction_features")
    return getattr(transaction_features, name)


__all__ = ["matrix", "FEATURE_NAMES", "event_level_features", "build_entity_windows"]
