"""Canonical Source and EMA loss APIs."""

from .common import (
    masked_bce, masked_gradient_error, masked_local_ssim_error, masked_mean,
    masked_smooth_l1, weighted_bce, weighted_l1,
)
from .ema import real_consistency_loss, stability_weights
from .cr import ContrastLoss, Vgg19
from .regional import regional_reconstruction_error
from .ssim import SSIM
from .source import (build_regional_reconstruction_criteria, charbonnier,
                     compute_physical_mask_losses)

__all__ = [
    "masked_mean", "masked_smooth_l1", "masked_bce", "weighted_l1", "weighted_bce",
    "masked_gradient_error", "masked_local_ssim_error", "charbonnier", "regional_reconstruction_error",
    "SSIM", "Vgg19", "ContrastLoss", "build_regional_reconstruction_criteria", "compute_physical_mask_losses",
    "stability_weights", "real_consistency_loss",
]
