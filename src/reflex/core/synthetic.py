"""Compatibility imports for the former synthetic core module.

Synthetic data generation and hidden evaluator labels are experiment concerns.
New code should import these types from ``reflex.experiments.synthetic_data``.
This module remains as a small bridge for existing local scripts.
"""

from reflex.experiments.synthetic_data import SyntheticDataset, generate_synthetic, write_dataset

__all__ = ["SyntheticDataset", "generate_synthetic", "write_dataset"]
