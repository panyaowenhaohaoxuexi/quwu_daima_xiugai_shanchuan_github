"""Canonical Source and EMA loss APIs."""

from .common import (
    masked_bce, masked_gradient_error, masked_local_ssim_error, masked_mean,
    masked_smooth_l1, weighted_bce, weighted_l1,
)
from .ema import real_consistency_loss, stability_weights
from .source import charbonnier, compute_physical_mask_losses

__all__ = [
    "masked_mean", "masked_smooth_l1", "masked_bce", "weighted_l1", "weighted_bce",
    "masked_gradient_error", "masked_local_ssim_error", "charbonnier", "compute_physical_mask_losses",
    "stability_weights", "real_consistency_loss",
]
