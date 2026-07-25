"""Common mask-aware and structural loss primitives."""

from .masked import masked_bce, masked_mean, masked_smooth_l1, weighted_bce, weighted_l1
from .structural import masked_gradient_error, masked_local_ssim_error

__all__ = [
    "masked_mean", "masked_smooth_l1", "masked_bce", "weighted_l1", "weighted_bce",
    "masked_gradient_error", "masked_local_ssim_error",
]
