"""Support-aware primitives for the formal fog-routed source objective."""

import torch
from torch.nn import functional as F


def _expanded(mask, value):
    return mask.detach().expand_as(value).to(dtype=value.dtype)


def masked_mean(value, mask, eps=1e-6):
    weight = _expanded(mask, value)
    denominator = weight.sum()
    if float(denominator.detach()) == 0.0:
        return value.sum() * 0.0
    return (value * weight).sum() / denominator.clamp_min(eps)


def masked_smooth_l1(prediction, target, mask, beta=1.0, eps=1e-6):
    if beta <= 0:
        raise ValueError("beta must be > 0")
    error = (prediction - target).abs()
    value = torch.where(error < beta, 0.5 * error.square() / beta, error - 0.5 * beta)
    return masked_mean(value, mask, eps)


def masked_bce(probability, target, mask, weight=None, eps=1e-6):
    probability = probability.clamp(eps, 1.0 - eps)
    target = target.detach()
    value = -(target * probability.log() + (1.0 - target) * (1.0 - probability).log())
    effective_mask = mask.detach()
    if weight is not None:
        effective_mask = effective_mask * weight.detach()
    return masked_mean(value, effective_mask, eps)


def binary_route_penalty(route_soft, validity_mask, eps=1e-6):
    return masked_mean(4.0 * route_soft * (1.0 - route_soft), validity_mask, eps)


def masked_gradient_error(prediction, target, support_mask, eps=1e-6):
    horizontal = (prediction[..., :, 1:] - prediction[..., :, :-1] -
                  (target[..., :, 1:] - target[..., :, :-1])).abs()
    vertical = (prediction[..., 1:, :] - prediction[..., :-1, :] -
                (target[..., 1:, :] - target[..., :-1, :])).abs()
    horizontal_mask = torch.minimum(support_mask[..., :, 1:], support_mask[..., :, :-1])
    vertical_mask = torch.minimum(support_mask[..., 1:, :], support_mask[..., :-1, :])
    return 0.5 * (masked_mean(horizontal, horizontal_mask, eps) + masked_mean(vertical, vertical_mask, eps))


def masked_local_ssim_error(prediction, target, support_mask, window_size=7,
                            min_valid_support=4, eps=1e-6):
    """SSIM error whose local moments never read outside ``support_mask``."""
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    if min_valid_support < 1:
        raise ValueError("min_valid_support must be >= 1")
    if prediction.shape != target.shape:
        raise ValueError("prediction and target must have matching shapes")
    weight = _expanded(support_mask, prediction)
    batch, channels, height, width = prediction.shape
    kernel = torch.ones(1, 1, window_size, window_size, device=prediction.device, dtype=prediction.dtype)

    def _window_sum(value):
        flat = value.reshape(batch * channels, 1, height, width)
        return F.conv2d(flat, kernel, padding=window_size // 2).reshape(batch, channels, height, width)

    support_sum = _window_sum(weight)
    denom = support_sum.clamp_min(eps)
    mean_x = _window_sum(prediction * weight) / denom
    mean_y = _window_sum(target * weight) / denom
    var_x = (_window_sum(prediction.square() * weight) / denom - mean_x.square()).clamp_min(0)
    var_y = (_window_sum(target.square() * weight) / denom - mean_y.square()).clamp_min(0)
    covariance = _window_sum(prediction * target * weight) / denom - mean_x * mean_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2) /
            ((mean_x.square() + mean_y.square() + c1) * (var_x + var_y + c2)).clamp_min(eps))
    valid_centers = (support_sum >= float(min_valid_support)).to(prediction.dtype) * weight
    return masked_mean(1.0 - ssim.clamp(-1, 1), valid_centers, eps)
