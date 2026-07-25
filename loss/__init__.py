"""Public canonical loss APIs; legacy modules remain importable by path."""

from .common import (
    masked_bce, masked_gradient_error, masked_local_ssim_error, masked_mean,
    masked_smooth_l1, weighted_bce, weighted_l1,
)
from .real import compute_adaptation_objective, real_consistency_loss, stability_weights
from .synthetic import compute_q, compute_source_objective

__all__ = [
    "masked_mean", "masked_smooth_l1", "masked_bce", "weighted_l1", "weighted_bce",
    "masked_gradient_error", "masked_local_ssim_error", "compute_q", "compute_source_objective",
    "stability_weights", "real_consistency_loss", "compute_adaptation_objective",
]
