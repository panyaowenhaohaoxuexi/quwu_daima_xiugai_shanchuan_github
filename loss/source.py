"""V2 physical completion-mask objective for Source training."""

import torch
from torch.nn import functional as F


def charbonnier(prediction, target, epsilon=1e-6):
    """Mean Charbonnier reconstruction error."""
    return torch.sqrt((prediction - target).square() + epsilon).mean()


def compute_physical_mask_losses(pred_clear, clear_rgb, density_map, density_gt, route_logits,
                                 completion_mask_gt, *, lambda_density=1.0, lambda_route=1.0,
                                 density_smooth_l1_beta=0.1):
    """Compute the complete V2 Source objective from one five-item batch."""
    if any(value.shape != density_map.shape for value in (density_gt, route_logits, completion_mask_gt)):
        raise ValueError("density, logits, and completion mask must share [B,1,H,W] shape")
    if pred_clear.shape != clear_rgb.shape:
        raise ValueError("pred_clear and clear_rgb must share shape")
    reconstruction = charbonnier(pred_clear, clear_rgb)
    density = F.smooth_l1_loss(density_map, density_gt, beta=density_smooth_l1_beta)
    positives = completion_mask_gt.sum()
    negatives = completion_mask_gt.numel() - positives
    pos_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 100.0).detach()
    route = F.binary_cross_entropy_with_logits(route_logits, completion_mask_gt, pos_weight=pos_weight)
    total = reconstruction + float(lambda_density) * density + float(lambda_route) * route
    return {"reconstruction": reconstruction, "density": density, "route": route,
            "total": total, "pos_weight": pos_weight}
