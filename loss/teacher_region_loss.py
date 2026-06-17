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


def _masked_mean(value, mask, eps=1e-6):
    denom = mask.sum().clamp_min(eps)
    if mask.sum() <= 0:
        return value.new_tensor(0.0)
    return (value * mask).sum() / denom


def _gradient_abs(x):
    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
    return dx, dy


def compute_teacher_region_loss(
    pred_clear,
    clear_gt,
    density_map,
    density_gt,
    mask_logits,
    mask_prob,
    mask_gt,
    hazy_vis_01=None,
    pred_raw=None,
    transported_rgb=None,
    semantic_ir=None,
    semantic_vis=None,
    proto_keys=None,
    proto_values=None,
    proto_attn=None,
    proto_assign=None,
    x_ir_01=None,
    binary_mask=None,
    lambda_rec=1.0,
    lambda_density=1.0,
    lambda_mask=1.0,
    lambda_ssim=0.0,
    lambda_cr=0.0,
    lambda_edge=0.0,
    lambda_align=0.0,
    lambda_comp=0.0,
    lambda_sparse=0.0,
    lambda_ir_tv=0.0,
    ir_tv_edge_lambda=10.0,
    ssim_module=None,
    contrast_module=None,
):
    """Compute synthetic Teacher losses.

    hazy_vis_01 is expected to be de-normalized to [0,1]; this function does
    not perform CLIP de-normalization.
    """
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

    if lambda_cr > 0:
        if hazy_vis_01 is None or contrast_module is None:
            raise ValueError("lambda_cr > 0 requires hazy_vis_01 and contrast_module.")
        hazy_vis_01 = hazy_vis_01.to(device=device, dtype=pred_clear.dtype)
        loss_cr = contrast_module(pred_clear, clear_gt, hazy_vis_01)
    else:
        loss_cr = _zero(device)

    if lambda_edge > 0:
        loss_edge = F.l1_loss(_sobel_edges(pred_clear), _sobel_edges(clear_gt).detach())
    else:
        loss_edge = _zero(device)

    # mask_gt is the stable supervision mask (M=1 completion, M=0 reliable).
    # binary_mask is reserved for model-internal forward color transport decisions.
    if lambda_comp > 0:
        mask_comp = F.interpolate(mask_gt, size=pred_clear.shape[2:], mode="nearest")
        mask_comp = (mask_comp >= 0.5).to(dtype=pred_clear.dtype)
        if mask_comp.sum() > 0:
            loss_comp = (mask_comp * torch.abs(pred_clear - clear_gt)).sum() / (
                mask_comp.sum() * pred_clear.shape[1] + 1e-6
            )
        else:
            loss_comp = _zero(device)
    else:
        loss_comp = _zero(device)

    if lambda_align > 0:
        if semantic_ir is None or semantic_vis is None:
            raise ValueError("lambda_align > 0 requires semantic_ir and semantic_vis.")
        semantic_ir = semantic_ir.to(device=device, dtype=pred_clear.dtype)
        semantic_vis = semantic_vis.to(device=device, dtype=pred_clear.dtype)
        mask_sem = F.interpolate(mask_gt, size=semantic_ir.shape[2:], mode="nearest")
        reliable_mask = (1.0 - (mask_sem >= 0.5).float()).to(dtype=pred_clear.dtype)
        if reliable_mask.sum() > 0:
            cos = (semantic_ir * semantic_vis).sum(dim=1, keepdim=True)
            loss_align = _masked_mean(1.0 - cos, reliable_mask)
        else:
            loss_align = _zero(device)
    else:
        loss_align = _zero(device)

    if lambda_sparse > 0:
        if proto_attn is None:
            raise ValueError("lambda_sparse > 0 requires proto_attn.")
        proto_attn = proto_attn.to(device=device, dtype=pred_clear.dtype)
        B, N, K = proto_attn.shape
        side_h = semantic_ir.shape[2] if semantic_ir is not None else int(round(N ** 0.5))
        side_w = semantic_ir.shape[3] if semantic_ir is not None else max(1, N // max(1, side_h))
        mask_sparse = F.interpolate(mask_gt, size=(side_h, side_w), mode="nearest")
        mask_sparse = (mask_sparse >= 0.5).float().flatten(2).squeeze(1).to(dtype=pred_clear.dtype)
        if mask_sparse.sum() > 0:
            entropy = -(proto_attn * torch.log(proto_attn.clamp_min(1e-6))).sum(dim=-1)
            loss_sparse = (entropy * mask_sparse).sum() / (mask_sparse.sum() + 1e-6)
        else:
            loss_sparse = _zero(device)
    else:
        loss_sparse = _zero(device)

    if lambda_ir_tv > 0:
        if x_ir_01 is None:
            raise ValueError("lambda_ir_tv > 0 requires x_ir_01.")
        x_ir_01 = x_ir_01.to(device=device, dtype=pred_clear.dtype)
        if x_ir_01.shape[2:] != pred_clear.shape[2:]:
            x_ir_01 = F.interpolate(x_ir_01, size=pred_clear.shape[2:], mode="bilinear", align_corners=False)
        mask_tv = F.interpolate(mask_gt, size=pred_clear.shape[2:], mode="nearest")
        mask_tv = (mask_tv >= 0.5).to(dtype=pred_clear.dtype)
        if mask_tv.sum() > 0:
            pred_dx, pred_dy = _gradient_abs(pred_clear)
            ir_dx, ir_dy = _gradient_abs(x_ir_01)
            mask_dx = mask_tv[:, :, :, 1:]
            mask_dy = mask_tv[:, :, 1:, :]
            grad_ir_dx = ir_dx.mean(dim=1, keepdim=True)
            grad_ir_dy = ir_dy.mean(dim=1, keepdim=True)
            weight_dx = torch.exp(-ir_tv_edge_lambda * grad_ir_dx)
            weight_dy = torch.exp(-ir_tv_edge_lambda * grad_ir_dy)
            tv_dx = (mask_dx * weight_dx * pred_dx.mean(dim=1, keepdim=True)).sum()
            tv_dy = (mask_dy * weight_dy * pred_dy.mean(dim=1, keepdim=True)).sum()
            denom = mask_dx.sum() + mask_dy.sum() + 1e-6
            loss_ir_tv = (tv_dx + tv_dy) / denom
        else:
            loss_ir_tv = _zero(device)
    else:
        loss_ir_tv = _zero(device)

    total = (
        lambda_rec * loss_rec
        + lambda_density * loss_density
        + lambda_mask * loss_mask
        + lambda_ssim * loss_ssim
        + lambda_cr * loss_cr
        + lambda_edge * loss_edge
        + lambda_align * loss_align
        + lambda_comp * loss_comp
        + lambda_sparse * loss_sparse
        + lambda_ir_tv * loss_ir_tv
    )

    return {
        "total": total,
        "rec": loss_rec,
        "density": loss_density,
        "mask": loss_mask,
        "ssim": loss_ssim,
        "cr": loss_cr,
        "edge": loss_edge,
        "align": loss_align,
        "comp": loss_comp,
        "sparse": loss_sparse,
        "ir_tv": loss_ir_tv,
    }
