"""Canonical Source-domain q, reconstruction, density, route and boundary losses."""

import torch
from torch.nn import functional as F

from .common import (
    masked_bce, masked_gradient_error, masked_local_ssim_error, masked_mean,
    masked_smooth_l1,
)


def local_supported_l1(prediction, target, support, window_size, min_valid_support):
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer")
    error = (prediction.detach() - target.detach()).abs().mean(dim=1, keepdim=True)
    mask = support.detach().clamp(0, 1)
    kernel = torch.ones(1, 1, window_size, window_size, device=error.device, dtype=error.dtype)
    numerator = F.conv2d(error * mask, kernel, padding=window_size // 2)
    denominator = F.conv2d(mask, kernel, padding=window_size // 2)
    return numerator / denominator.clamp_min(1e-6), (denominator >= float(min_valid_support)).to(error.dtype) * mask


def local_supported_gradient_error(prediction, target, support, window_size, min_valid_support):
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
    return local_supported_l1(edge_error / edge_weight.clamp_min(1e-6), torch.zeros_like(mask),
                              (edge_weight > 0).to(mask.dtype) * mask, window_size, min_valid_support)


def local_supported_ssim_error(prediction, target, support, window_size, min_valid_support):
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


def compute_q(clear_rgb, pred_fusion, pred_completion, omega_support, temperature, window_size=1,
              min_valid_support=1, l1_weight=1.0, gradient_weight=0.0, ssim_weight=0.0):
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    support = omega_support.detach()
    fusion_l1, fusion_valid = local_supported_l1(pred_fusion, clear_rgb, support, window_size, min_valid_support)
    completion_l1, completion_valid = local_supported_l1(pred_completion, clear_rgb, support, window_size, min_valid_support)
    fusion_weighted, completion_weighted = l1_weight * fusion_l1 * fusion_valid, l1_weight * completion_l1 * completion_valid
    fusion_available, completion_available = l1_weight * fusion_valid, l1_weight * completion_valid
    if gradient_weight:
        fusion_gradient, fusion_gradient_valid = local_supported_gradient_error(pred_fusion.detach(), clear_rgb.detach(), support, window_size, min_valid_support)
        completion_gradient, completion_gradient_valid = local_supported_gradient_error(pred_completion.detach(), clear_rgb.detach(), support, window_size, min_valid_support)
        fusion_weighted += gradient_weight * fusion_gradient * fusion_gradient_valid
        completion_weighted += gradient_weight * completion_gradient * completion_gradient_valid
        fusion_available += gradient_weight * fusion_gradient_valid
        completion_available += gradient_weight * completion_gradient_valid
    if ssim_weight:
        fusion_ssim, fusion_ssim_valid = local_supported_ssim_error(pred_fusion, clear_rgb, support, window_size, min_valid_support)
        completion_ssim, completion_ssim_valid = local_supported_ssim_error(pred_completion, clear_rgb, support, window_size, min_valid_support)
        fusion_weighted += ssim_weight * fusion_ssim * fusion_ssim_valid
        completion_weighted += ssim_weight * completion_ssim * completion_ssim_valid
        fusion_available += ssim_weight * fusion_ssim_valid
        completion_available += ssim_weight * completion_ssim_valid
    valid = ((fusion_available > 0) * (completion_available > 0) * support).detach()
    q = torch.sigmoid((fusion_weighted / fusion_available.clamp_min(1e-6) -
                       completion_weighted / completion_available.clamp_min(1e-6)) / float(temperature))
    return (q * valid).detach(), valid


def reconstruction_error(prediction, target, weight, *, l1_weight, gradient_weight, ssim_weight,
                         ssim_window, min_valid_support):
    return (l1_weight * masked_mean((prediction - target).abs(), weight) +
            gradient_weight * masked_gradient_error(prediction, target, weight) +
            ssim_weight * masked_local_ssim_error(prediction, target, weight, ssim_window, min_valid_support))


def compute_source_objective(pred_clear, clear_rgb, density_map, density_gt, route_soft, boundary_map,
                             q, omega_support, omega_weight=None, density_beta=0.1, lambda_global=1.0,
                             lambda_fuse=1.0, lambda_comp=1.0, lambda_boundary=1.0, lambda_density=1.0,
                             lambda_route=1.0, lambda_binary=1.0, lambda_router=1.0,
                             *, hazy_rgb=None, global_ssim_criterion=None, global_contrast_criterion=None,
                             global_l1_weight=0.8, global_ssim_weight=0.2, global_contrast_weight=0.05,
                             region_l1_weight=1.0, region_gradient_weight=0.2, region_ssim_weight=0.2,
                             ssim_window=7, min_valid_support=4):
    if global_contrast_weight > 0 and (hazy_rgb is None or global_contrast_criterion is None):
        raise ValueError("CoA global contrast loss requires hazy_rgb and global_contrast_criterion")
    validity = density_gt.new_ones(density_gt.shape)
    route, boundary = route_soft.detach(), boundary_map.detach()
    kwargs = {"l1_weight": region_l1_weight, "gradient_weight": region_gradient_weight,
              "ssim_weight": region_ssim_weight, "ssim_window": ssim_window,
              "min_valid_support": min_valid_support}
    global_l1 = masked_mean((pred_clear - clear_rgb).abs(), validity)
    if global_ssim_criterion is None:
        from .coa_reconstruction import CoASSIM
        global_ssim_criterion = CoASSIM()
    global_ssim = 1.0 - global_ssim_criterion(pred_clear, clear_rgb)
    global_contrast = (global_contrast_criterion(pred_clear, clear_rgb, hazy_rgb)
                       if global_contrast_weight > 0 else pred_clear.new_zeros(()))
    global_loss = (global_l1_weight * global_l1 + global_ssim_weight * global_ssim +
                   global_contrast_weight * global_contrast)
    fuse_loss = reconstruction_error(pred_clear, clear_rgb, (1 - route) * validity, **kwargs)
    comp_loss = reconstruction_error(pred_clear, clear_rgb, route * validity, **kwargs)
    boundary_support = boundary * validity
    boundary_value = reconstruction_error(pred_clear, clear_rgb, boundary_support, **kwargs)
    density_value = masked_smooth_l1(density_map, density_gt, validity, density_beta)
    route_value = masked_bce(route_soft, q.detach(), omega_support.detach(), omega_weight)
    binary_value = masked_mean(4.0 * route_soft * (1.0 - route_soft), validity)
    router_total = lambda_density * density_value + lambda_route * route_value + lambda_binary * binary_value
    total = (lambda_global * global_loss + lambda_fuse * fuse_loss + lambda_comp * comp_loss +
             lambda_boundary * boundary_value + lambda_router * router_total)
    return {"total": total, "global": global_loss, "global_l1": global_l1, "global_ssim": global_ssim,
            "global_contrast": global_contrast, "fuse": fuse_loss, "comp": comp_loss,
            "boundary": boundary_value, "density": density_value, "route": route_value, "binary": binary_value,
            "region_l1_weight": region_l1_weight, "region_ssim_weight": region_ssim_weight,
            "region_gradient_weight": region_gradient_weight}
