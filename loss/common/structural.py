"""Support-aware gradient and local-SSIM reconstruction errors."""

import torch
from torch.nn import functional as F

from .masked import _expanded, masked_mean


def masked_gradient_error(prediction: torch.Tensor, target: torch.Tensor,
                          support_mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    horizontal = (prediction[..., :, 1:] - prediction[..., :, :-1] -
                  (target[..., :, 1:] - target[..., :, :-1])).abs()
    vertical = (prediction[..., 1:, :] - prediction[..., :-1, :] -
                (target[..., 1:, :] - target[..., :-1, :])).abs()
    horizontal_mask = torch.minimum(support_mask[..., :, 1:], support_mask[..., :, :-1])
    vertical_mask = torch.minimum(support_mask[..., 1:, :], support_mask[..., :-1, :])
    return 0.5 * (masked_mean(horizontal, horizontal_mask, eps) + masked_mean(vertical, vertical_mask, eps))


def masked_local_ssim_error(prediction: torch.Tensor, target: torch.Tensor, support_mask: torch.Tensor,
                            window_size: int = 7, min_valid_support: int = 4,
                            eps: float = 1e-6) -> torch.Tensor:
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

    def _window_sum(value: torch.Tensor) -> torch.Tensor:
        flat = value.reshape(batch * channels, 1, height, width)
        return F.conv2d(flat, kernel, padding=window_size // 2).reshape(batch, channels, height, width)

    support_sum = _window_sum(weight)
    denominator = support_sum.clamp_min(eps)
    mean_x = _window_sum(prediction * weight) / denominator
    mean_y = _window_sum(target * weight) / denominator
    var_x = (_window_sum(prediction.square() * weight) / denominator - mean_x.square()).clamp_min(0)
    var_y = (_window_sum(target.square() * weight) / denominator - mean_y.square()).clamp_min(0)
    covariance = _window_sum(prediction * target * weight) / denominator - mean_x * mean_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2) /
            ((mean_x.square() + mean_y.square() + c1) * (var_x + var_y + c2)).clamp_min(eps))
    valid_centers = (support_sum >= float(min_valid_support)).to(prediction.dtype) * weight
    return masked_mean(1.0 - ssim.clamp(-1, 1), valid_centers, eps)


__all__ = ["masked_gradient_error", "masked_local_ssim_error"]
