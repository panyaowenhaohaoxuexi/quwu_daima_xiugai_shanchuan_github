"""Canonical synthetic-domain loss formulas."""

from .counterfactual import compute_q
from .objective import compute_source_objective

__all__ = ["compute_q", "compute_source_objective"]
