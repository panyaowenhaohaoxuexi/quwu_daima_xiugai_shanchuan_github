"""Composite synthetic reconstruction error."""

import torch

from loss.common.masked import masked_mean
from loss.common.structural import masked_gradient_error, masked_local_ssim_error


def reconstruction_error(prediction: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, *,
                         l1_weight: float, gradient_weight: float, ssim_weight: float,
                         ssim_window: int, min_valid_support: int) -> torch.Tensor:
    l1 = masked_mean((prediction - target).abs(), weight)
    gradient = masked_gradient_error(prediction, target, weight)
    ssim = masked_local_ssim_error(prediction, target, weight, ssim_window, min_valid_support)
    return l1_weight * l1 + gradient_weight * gradient + ssim_weight * ssim


__all__ = ["reconstruction_error"]
