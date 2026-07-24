"""Shared-context counterfactual helpers for source route supervision."""

import torch
from torch.nn import functional as F


def detached_context(context):
    """Detach every tensor in a nested context without rerunning an encoder."""
    if torch.is_tensor(context):
        return context.detach()
    if isinstance(context, dict):
        return {key: detached_context(value) for key, value in context.items()}
    if isinstance(context, tuple):
        return tuple(detached_context(value) for value in context)
    if isinstance(context, list):
        return [detached_context(value) for value in context]
    return context


def gather_context(context, owner_indices):
    """Gather only a counterfactual chunk's owners from detached shared context.

    This deliberately indexes existing encoded features instead of expanding
    every pyramid tensor to ``B x K`` up front.
    """
    if not isinstance(context, dict) or "density_map" not in context:
        raise ValueError("context must be the formal encode_context dictionary")
    owners = owner_indices.detach().long()
    batch_size = context["density_map"].shape[0]

    def _gather(value):
        if torch.is_tensor(value):
            if value.ndim > 0 and value.shape[0] == batch_size:
                return value.index_select(0, owners)
            return value
        if isinstance(value, dict):
            return {key: _gather(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(_gather(item) for item in value)
        if isinstance(value, list):
            return [_gather(item) for item in value]
        return value

    return _gather(context)


def build_counterfactual_routes(context, omega_support):
    """Build the two explicit route overrides over the original-size Omega support."""
    support = omega_support.detach().clamp(0, 1)
    return (
        {
            "route_override_value": torch.zeros_like(support),
            "route_override_mask": support,
            "memory_exclude_mask": torch.zeros_like(support),
        },
        {
            "route_override_value": torch.ones_like(support),
            "route_override_mask": support,
            "memory_exclude_mask": support,
        },
    )


def run_counterfactual_pair(model, context, omega_support, route_mode="hard", boundary_mode="soft"):
    """Decode fusion/completion candidates without rerunning shared encoders."""
    context = detached_context(context)
    fusion_args, completion_args = build_counterfactual_routes(context, omega_support)
    with torch.no_grad():
        fusion = model.decode_with_route(context, route_mode=route_mode, boundary_mode=boundary_mode, **fusion_args)
        completion = model.decode_with_route(context, route_mode=route_mode, boundary_mode=boundary_mode, **completion_args)
    return fusion, completion


def run_counterfactual_chunks(model, context, owner_indices, omega_support,
                              chunk_size, route_mode="hard", boundary_mode="soft"):
    """Run candidate decoding in bounded owner-index chunks.

    ``context`` is the one shared main-forward context.  Only the feature rows
    required by a chunk are gathered; no encoder method is called here.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    owners = owner_indices.detach().long()
    support = omega_support.detach()
    if owners.ndim != 1 or owners.numel() != support.shape[0]:
        raise ValueError("owner_indices must match the number of Omega supports")
    fusion_predictions, completion_predictions = [], []
    for start in range(0, owners.numel(), chunk_size):
        end = min(start + chunk_size, owners.numel())
        chunk_context = gather_context(detached_context(context), owners[start:end])
        fusion, completion = run_counterfactual_pair(
            model, chunk_context, support[start:end], route_mode=route_mode, boundary_mode=boundary_mode
        )
        fusion_predictions.append(fusion["pred_clear"])
        completion_predictions.append(completion["pred_clear"])
    if not fusion_predictions:
        empty = support.new_zeros((0, 3, *support.shape[-2:]))
        return {"pred_clear": empty}, {"pred_clear": empty}
    return (
        {"pred_clear": torch.cat(fusion_predictions, dim=0)},
        {"pred_clear": torch.cat(completion_predictions, dim=0)},
    )


def _local_supported_l1(prediction, target, support, window_size, min_valid_support):
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


def _local_supported_gradient(prediction, target, support, window_size, min_valid_support):
    mask = support.detach().clamp(0, 1)
    horizontal = ((prediction[..., :, 1:] - prediction[..., :, :-1]) -
                  (target[..., :, 1:] - target[..., :, :-1])).abs().mean(dim=1, keepdim=True)
    vertical = ((prediction[..., 1:, :] - prediction[..., :-1, :]) -
                (target[..., 1:, :] - target[..., :-1, :])).abs().mean(dim=1, keepdim=True)
    horizontal_mask = torch.minimum(mask[..., :, 1:], mask[..., :, :-1])
    vertical_mask = torch.minimum(mask[..., 1:, :], mask[..., :-1, :])
    # Place each valid edge at both endpoints so later local averaging remains
    # inside support and no outside pixel can contribute.
    edge_error = torch.zeros_like(mask)
    edge_weight = torch.zeros_like(mask)
    edge_error[..., :, :-1] += horizontal * horizontal_mask
    edge_error[..., :, 1:] += horizontal * horizontal_mask
    edge_weight[..., :, :-1] += horizontal_mask
    edge_weight[..., :, 1:] += horizontal_mask
    edge_error[..., :-1, :] += vertical * vertical_mask
    edge_error[..., 1:, :] += vertical * vertical_mask
    edge_weight[..., :-1, :] += vertical_mask
    edge_weight[..., 1:, :] += vertical_mask
    point_error = edge_error / edge_weight.clamp_min(1e-6)
    return _local_supported_l1(point_error, torch.zeros_like(point_error),
                               (edge_weight > 0).to(mask.dtype) * mask,
                               window_size, min_valid_support)


def _local_supported_ssim_error(prediction, target, support, window_size, min_valid_support):
    mask = support.detach().clamp(0, 1)
    batch, channels, height, width = prediction.shape
    kernel = torch.ones(1, 1, window_size, window_size, device=prediction.device, dtype=prediction.dtype)

    def local_sum(value):
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


def compute_q(clear_rgb, pred_fusion, pred_completion, omega_support, temperature,
              window_size=1, min_valid_support=1, l1_weight=1.0, gradient_weight=0.0,
              ssim_weight=0.0):
    """Detached q where one means completion has lower *local* supported error."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    support = omega_support.detach()
    error_fusion, valid_fusion = _local_supported_l1(
        pred_fusion, clear_rgb, support, window_size, min_valid_support
    )
    error_completion, valid_completion = _local_supported_l1(
        pred_completion, clear_rgb, support, window_size, min_valid_support
    )
    fusion_weighted = l1_weight * error_fusion * valid_fusion
    completion_weighted = l1_weight * error_completion * valid_completion
    fusion_available = l1_weight * valid_fusion
    completion_available = l1_weight * valid_completion
    if gradient_weight:
        gradient_fusion, gradient_valid_fusion = _local_supported_gradient(
            pred_fusion.detach(), clear_rgb.detach(), support, window_size, min_valid_support
        )
        gradient_completion, gradient_valid_completion = _local_supported_gradient(
            pred_completion.detach(), clear_rgb.detach(), support, window_size, min_valid_support
        )
        fusion_weighted = fusion_weighted + gradient_weight * gradient_fusion * gradient_valid_fusion
        completion_weighted = completion_weighted + gradient_weight * gradient_completion * gradient_valid_completion
        fusion_available = fusion_available + gradient_weight * gradient_valid_fusion
        completion_available = completion_available + gradient_weight * gradient_valid_completion
    if ssim_weight:
        ssim_fusion, ssim_valid_fusion = _local_supported_ssim_error(
            pred_fusion, clear_rgb, support, window_size, min_valid_support
        )
        ssim_completion, ssim_valid_completion = _local_supported_ssim_error(
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
