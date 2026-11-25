"""Shared immutable base types used by the REFLEX domain models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


class FrozenDict(dict):
    """A JSON-serializable mapping used for immutable record contents.

    REFLEX records often need nested dictionaries because event attributes and
    observation values are intentionally flexible. Keeping the mapping type
    familiar makes those records easy to serialize while rejecting the usual
    in-place dictionary mutators.
    """

    def __init__(self, values: dict[str, Any] | None = None):
        dict.__init__(self, values or {})

    def _immutable(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("record contents are immutable")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _immutable


def _deep_freeze(value: Any) -> Any:
    """Freeze nested containers so a saved domain record cannot be edited."""
    if isinstance(value, FrozenDict):
        return value
    if isinstance(value, dict):
        return FrozenDict({key: _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_deep_freeze(item) for item in value)
    return value


class FrozenModel(BaseModel):
    """Base class for validated records with immutable fields and containers.

    Pydantic validates the declared shape and blocks field reassignment. The
    after-validator also walks nested dictionaries, lists, and sets so callers
    cannot mutate a record indirectly after it has been written or referenced
    as evidence. Tuples and scalar values already have immutable semantics and
    are retained as-is.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def freeze_nested_values(self) -> "FrozenModel":
        for name, value in self.__dict__.items():
            frozen = _deep_freeze(value)
            if frozen is not value:
                object.__setattr__(self, name, frozen)
        return self
