"""Small formal `[0,1]` image metrics used only after model output."""

from __future__ import annotations

import torch


def psnr(prediction, target, eps=1e-12):
    mse = (prediction - target).square().mean()
    return -10.0 * torch.log10(mse.clamp_min(eps))


def ssim_global(prediction, target, eps=1e-12):
    """Global SSIM diagnostic (losses use the support-aware local operator)."""
    mean_x, mean_y = prediction.mean(), target.mean()
    var_x = (prediction - mean_x).square().mean()
    var_y = (target - mean_y).square().mean()
    covariance = ((prediction - mean_x) * (target - mean_y)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mean_x * mean_y + c1) * (2 * covariance + c2) /
            ((mean_x.square() + mean_y.square() + c1) * (var_x + var_y + c2)).clamp_min(eps))
