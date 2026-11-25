"""Shared immutable base types used by the REFLEX domain models."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


class FrozenDict(dict):
    """A JSON-compatible dictionary that rejects in-place updates."""

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
    """Base for validated REFLEX records with immutable fields and contents."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    @model_validator(mode="after")
    def freeze_nested_values(self) -> "FrozenModel":
        for name, value in self.__dict__.items():
            frozen = _deep_freeze(value)
            if frozen is not value:
                object.__setattr__(self, name, frozen)
        return self
