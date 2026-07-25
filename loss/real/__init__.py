"""Canonical real-domain adaptation losses."""

from .consistency import real_consistency_loss, stability_weights, weighted_bce, weighted_l1
from .objective import compute_adaptation_objective

__all__ = [
    "stability_weights", "weighted_l1", "weighted_bce", "real_consistency_loss",
    "compute_adaptation_objective",
]
