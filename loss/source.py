"""V2 physical completion-mask objective for Source training."""

import torch
from torch.nn import functional as F

from .common import masked_mean
from .cr import ContrastLoss
from .regional import regional_reconstruction_error
from .ssim import SSIM


def charbonnier(prediction, target, epsilon=1e-6):
    """Mean Charbonnier reconstruction error."""
    return torch.sqrt((prediction - target).square() + epsilon).mean()


def build_regional_reconstruction_criteria(device):
    """Create the original global SSIM and VGG contrast terms once per process."""
    return SSIM().to(device), ContrastLoss().to(device)


def compute_physical_mask_losses(pred_clear, clear_rgb, density_map, density_gt, route_logits,
                                 completion_mask_gt, *, route_for_reconstruction, boundary_map,
                                 hazy_rgb, global_ssim_criterion, global_contrast_criterion,
                                 lambda_density=1.0, lambda_route=1.0,
                                 density_smooth_l1_beta=0.1,
                                 lambda_global=1.0, lambda_fuse=1.0, lambda_comp=1.0,
                                 lambda_boundary=1.0, global_l1_weight=0.8,
                                 global_ssim_weight=0.2, global_contrast_weight=0.05,
                                 region_l1_weight=1.0, region_gradient_weight=0.2,
                                 region_ssim_weight=0.2, reconstruction_ssim_window=7,
                                 reconstruction_min_valid_support=4):
    """V2 mask supervision plus the original global and route-region reconstruction terms."""
    if any(value.shape != density_map.shape for value in (density_gt, route_logits, completion_mask_gt)):
        raise ValueError("density, logits, and completion mask must share [B,1,H,W] shape")
    if pred_clear.shape != clear_rgb.shape:
        raise ValueError("pred_clear and clear_rgb must share shape")
    if any(value.shape != density_map.shape for value in (route_for_reconstruction, boundary_map)):
        raise ValueError("route and boundary maps must share density shape")
    if hazy_rgb.shape != clear_rgb.shape:
        raise ValueError("hazy_rgb and clear_rgb must share shape")
    validity = torch.ones_like(density_map)
    route = route_for_reconstruction.detach().clamp(0, 1)
    boundary = boundary_map.detach().clamp(0, 1)
    global_l1 = masked_mean((pred_clear - clear_rgb).abs(), validity)
    global_ssim = 1.0 - global_ssim_criterion(pred_clear, clear_rgb)
    global_contrast = global_contrast_criterion(pred_clear, clear_rgb, hazy_rgb)
    global_loss = (float(global_l1_weight) * global_l1 + float(global_ssim_weight) * global_ssim +
                   float(global_contrast_weight) * global_contrast)
    regional_kwargs = {"l1_weight": region_l1_weight, "gradient_weight": region_gradient_weight,
                       "ssim_weight": region_ssim_weight, "ssim_window": reconstruction_ssim_window,
                       "min_valid_support": reconstruction_min_valid_support}
    fuse = regional_reconstruction_error(pred_clear, clear_rgb, 1.0 - route, **regional_kwargs)
    comp = regional_reconstruction_error(pred_clear, clear_rgb, route, **regional_kwargs)
    boundary_value = regional_reconstruction_error(pred_clear, clear_rgb, boundary, **regional_kwargs)
    reconstruction = (float(lambda_global) * global_loss + float(lambda_fuse) * fuse +
                      float(lambda_comp) * comp + float(lambda_boundary) * boundary_value)
    density = F.smooth_l1_loss(density_map, density_gt, beta=density_smooth_l1_beta)
    positives = completion_mask_gt.sum()
    negatives = completion_mask_gt.numel() - positives
    pos_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 100.0).detach()
    route = F.binary_cross_entropy_with_logits(route_logits, completion_mask_gt, pos_weight=pos_weight)
    total = reconstruction + float(lambda_density) * density + float(lambda_route) * route
    return {"reconstruction": reconstruction, "global": global_loss, "global_l1": global_l1,
            "global_ssim": global_ssim, "global_contrast": global_contrast, "fuse": fuse,
            "comp": comp, "boundary": boundary_value, "density": density, "route": route,
            "total": total, "pos_weight": pos_weight}
