import torch
import torch.nn.functional as F

try:
    from .SSIM import SSIM
except Exception:
    from SSIM import SSIM


def _zero(device):
    return torch.tensor(0.0, device=device)


def _dice_loss(prob, target, eps=1e-6):
    prob = prob.clamp(eps, 1.0 - eps)
    target = target.float()
    dims = tuple(range(1, prob.dim()))
    inter = (prob * target).sum(dim=dims)
    denom = prob.sum(dim=dims) + target.sum(dim=dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def _sobel_edges(x):
    channels = x.shape[1]
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    kx = kx.repeat(channels, 1, 1, 1)
    ky = ky.repeat(channels, 1, 1, 1)
    gx = F.conv2d(x, kx, padding=1, groups=channels)
    gy = F.conv2d(x, ky, padding=1, groups=channels)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def compute_teacher_region_loss(
    pred_clear,
    clear_gt,
    density_map,
    density_gt,
    mask_logits,
    mask_prob,
    mask_gt,
    lambda_rec=1.0,
    lambda_density=1.0,
    lambda_mask=1.0,
    lambda_ssim=0.0,
    lambda_edge=0.0,
    ssim_module=None,
):
    device = pred_clear.device
    clear_gt = clear_gt.to(device=device, dtype=pred_clear.dtype)
    density_gt = density_gt.to(device=device, dtype=density_map.dtype)
    mask_gt = mask_gt.to(device=device, dtype=mask_prob.dtype)

    if density_gt.shape[2:] != density_map.shape[2:]:
        density_gt = F.interpolate(density_gt, size=density_map.shape[2:], mode="bilinear", align_corners=False)
    density_gt = density_gt.clamp(0.0, 1.0)

    if mask_gt.shape[2:] != mask_prob.shape[2:]:
        mask_gt = F.interpolate(mask_gt, size=mask_prob.shape[2:], mode="nearest")
    mask_gt = (mask_gt >= 0.5).float()

    loss_rec = F.l1_loss(pred_clear, clear_gt) if lambda_rec > 0 else _zero(device)
    loss_density = F.l1_loss(density_map, density_gt) if lambda_density > 0 else _zero(device)

    if lambda_mask > 0:
        if mask_logits is not None:
            mask_logits = mask_logits.to(device=device, dtype=mask_prob.dtype)
            if mask_logits.shape[2:] != mask_gt.shape[2:]:
                mask_gt_for_bce = F.interpolate(mask_gt, size=mask_logits.shape[2:], mode="nearest")
                mask_gt_for_bce = (mask_gt_for_bce >= 0.5).float()
            else:
                mask_gt_for_bce = mask_gt
            bce = F.binary_cross_entropy_with_logits(mask_logits, mask_gt_for_bce)
        else:
            bce = F.binary_cross_entropy(mask_prob.clamp(1e-6, 1.0 - 1e-6), mask_gt)
        loss_mask = bce + _dice_loss(mask_prob, mask_gt)
    else:
        loss_mask = _zero(device)

    if lambda_ssim > 0:
        if ssim_module is None:
            ssim_module = SSIM().to(device)
        loss_ssim = 1.0 - ssim_module(pred_clear, clear_gt)
    else:
        loss_ssim = _zero(device)

    if lambda_edge > 0:
        loss_edge = F.l1_loss(_sobel_edges(pred_clear), _sobel_edges(clear_gt).detach())
    else:
        loss_edge = _zero(device)

    total = (
        lambda_rec * loss_rec
        + lambda_density * loss_density
        + lambda_mask * loss_mask
        + lambda_ssim * loss_ssim
        + lambda_edge * loss_edge
    )

    return {
        "total": total,
        "rec": loss_rec,
        "density": loss_density,
        "mask": loss_mask,
        "ssim": loss_ssim,
        "edge": loss_edge,
    }
