"""Detached local counterfactual error and route-target formulas."""

import torch
from torch.nn import functional as F


def local_supported_l1(prediction: torch.Tensor, target: torch.Tensor, support: torch.Tensor,
                       window_size: int, min_valid_support: int) -> tuple[torch.Tensor, torch.Tensor]:
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    error = (prediction.detach() - target.detach()).abs().mean(dim=1, keepdim=True)
    mask = support.detach().clamp(0, 1)
    kernel = torch.ones(1, 1, window_size, window_size, device=error.device, dtype=error.dtype)
    numerator = F.conv2d(error * mask, kernel, padding=window_size // 2)
    denominator = F.conv2d(mask, kernel, padding=window_size // 2)
    local = numerator / denominator.clamp_min(1e-6)
    valid = (denominator >= float(min_valid_support)).to(error.dtype) * mask
    return local, valid


def local_supported_gradient_error(prediction: torch.Tensor, target: torch.Tensor, support: torch.Tensor,
                                   window_size: int, min_valid_support: int) -> tuple[torch.Tensor, torch.Tensor]:
    mask = support.detach().clamp(0, 1)
    horizontal = ((prediction[..., :, 1:] - prediction[..., :, :-1]) -
                  (target[..., :, 1:] - target[..., :, :-1])).abs().mean(dim=1, keepdim=True)
    vertical = ((prediction[..., 1:, :] - prediction[..., :-1, :]) -
                (target[..., 1:, :] - target[..., :-1, :])).abs().mean(dim=1, keepdim=True)
    horizontal_mask = torch.minimum(mask[..., :, 1:], mask[..., :, :-1])
    vertical_mask = torch.minimum(mask[..., 1:, :], mask[..., :-1, :])
    edge_error, edge_weight = torch.zeros_like(mask), torch.zeros_like(mask)
    edge_error[..., :, :-1] += horizontal * horizontal_mask
    edge_error[..., :, 1:] += horizontal * horizontal_mask
    edge_weight[..., :, :-1] += horizontal_mask
    edge_weight[..., :, 1:] += horizontal_mask
    edge_error[..., :-1, :] += vertical * vertical_mask
    edge_error[..., 1:, :] += vertical * vertical_mask
    edge_weight[..., :-1, :] += vertical_mask
    edge_weight[..., 1:, :] += vertical_mask
    point_error = edge_error / edge_weight.clamp_min(1e-6)
    return local_supported_l1(point_error, torch.zeros_like(point_error),
                              (edge_weight > 0).to(mask.dtype) * mask,
                              window_size, min_valid_support)


def local_supported_ssim_error(prediction: torch.Tensor, target: torch.Tensor, support: torch.Tensor,
                               window_size: int, min_valid_support: int) -> tuple[torch.Tensor, torch.Tensor]:
    mask = support.detach().clamp(0, 1)
    batch, channels, height, width = prediction.shape
    kernel = torch.ones(1, 1, window_size, window_size, device=prediction.device, dtype=prediction.dtype)

    def local_sum(value: torch.Tensor) -> torch.Tensor:
        return F.conv2d(value.reshape(batch * channels, 1, height, width), kernel,
                        padding=window_size // 2).reshape(batch, channels, height, width)

    weight = mask.expand_as(prediction)
    weight_sum = local_sum(weight).clamp_min(1e-6)
    mean_x = local_sum(prediction.detach() * weight) / weight_sum
    mean_y = local_sum(target.detach() * weight) / weight_sum
    var_x = (local_sum(prediction.detach().square() * weight) / weight_sum - mean_x.square()).clamp_min(0)
    var_y = (local_sum(target.detach().square() * weight) / weight_sum - mean_y.square()).clamp_min(0)
    covariance = local_sum(prediction.detach() * target.detach() * weight) / weight_sum - mean_x * mean_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2) /
            ((mean_x.square() + mean_y.square() + c1) * (var_x + var_y + c2)).clamp_min(1e-6))
    support_count = F.conv2d(mask, kernel, padding=window_size // 2)
    valid = (support_count >= float(min_valid_support)).to(mask.dtype) * mask
    return (1.0 - ssim.clamp(-1, 1)).mean(dim=1, keepdim=True), valid


def compute_q(clear_rgb: torch.Tensor, pred_fusion: torch.Tensor, pred_completion: torch.Tensor,
              omega_support: torch.Tensor, temperature: float, window_size: int = 1,
              min_valid_support: int = 1, l1_weight: float = 1.0,
              gradient_weight: float = 0.0, ssim_weight: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Detached q where one means completion has lower local supported error."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    support = omega_support.detach()
    error_fusion, valid_fusion = local_supported_l1(pred_fusion, clear_rgb, support, window_size, min_valid_support)
    error_completion, valid_completion = local_supported_l1(
        pred_completion, clear_rgb, support, window_size, min_valid_support
    )
    fusion_weighted = l1_weight * error_fusion * valid_fusion
    completion_weighted = l1_weight * error_completion * valid_completion
    fusion_available, completion_available = l1_weight * valid_fusion, l1_weight * valid_completion
    if gradient_weight:
        gradient_fusion, gradient_valid_fusion = local_supported_gradient_error(
            pred_fusion.detach(), clear_rgb.detach(), support, window_size, min_valid_support
        )
        gradient_completion, gradient_valid_completion = local_supported_gradient_error(
            pred_completion.detach(), clear_rgb.detach(), support, window_size, min_valid_support
        )
        fusion_weighted = fusion_weighted + gradient_weight * gradient_fusion * gradient_valid_fusion
        completion_weighted = completion_weighted + gradient_weight * gradient_completion * gradient_valid_completion
        fusion_available = fusion_available + gradient_weight * gradient_valid_fusion
        completion_available = completion_available + gradient_weight * gradient_valid_completion
    if ssim_weight:
        ssim_fusion, ssim_valid_fusion = local_supported_ssim_error(
            pred_fusion, clear_rgb, support, window_size, min_valid_support
        )
        ssim_completion, ssim_valid_completion = local_supported_ssim_error(
            pred_completion, clear_rgb, support, window_size, min_valid_support
        )
        fusion_weighted = fusion_weighted + ssim_weight * ssim_fusion * ssim_valid_fusion
        completion_weighted = completion_weighted + ssim_weight * ssim_completion * ssim_valid_completion
        fusion_available = fusion_available + ssim_weight * ssim_valid_fusion
        completion_available = completion_available + ssim_weight * ssim_valid_completion
    error_fusion = fusion_weighted / fusion_available.clamp_min(1e-6)
    error_completion = completion_weighted / completion_available.clamp_min(1e-6)
    valid = ((fusion_available > 0) * (completion_available > 0) * support).detach()
    q = torch.sigmoid((error_fusion - error_completion) / float(temperature))
    return (q * valid).detach(), valid


__all__ = [
    "local_supported_l1", "local_supported_gradient_error", "local_supported_ssim_error", "compute_q",
]
