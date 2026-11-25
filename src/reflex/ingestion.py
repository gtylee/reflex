"""Compatibility imports for the former root-level news event adapter.

New code should import these dataset-specific types from
``reflex.experiments.news_adapters``. The reusable core accepts normalized
``Event`` objects and does not prescribe a source format.
"""

from reflex.experiments.news_adapters import GenericNewsAdapter, NewsStoryAdapter

__all__ = ["NewsStoryAdapter", "GenericNewsAdapter"]
