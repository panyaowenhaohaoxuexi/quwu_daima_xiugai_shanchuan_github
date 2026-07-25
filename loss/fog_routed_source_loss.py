"""Compatibility exports for the pre-layered formal source loss module."""

from loss.common.masked import masked_bce, masked_mean, masked_smooth_l1
from loss.common.structural import masked_gradient_error, masked_local_ssim_error
from loss.synthetic.routing import binary_route_penalty

__all__ = [
    "masked_mean", "masked_smooth_l1", "masked_bce", "binary_route_penalty",
    "masked_gradient_error", "masked_local_ssim_error",
]
