import torch

from loss.teacher_region_loss import compute_teacher_region_loss


def _base_tensors(mask_mode):
    torch.manual_seed(11)
    b, h, w = 2, 16, 16
    hf, wf, d, k = 4, 4, 8, 3
    pred_clear = torch.rand(b, 3, h, w)
    clear_gt = torch.rand(b, 3, h, w)
    density_map = torch.rand(b, 1, h, w)
    density_gt = torch.rand(b, 1, h, w)
    mask_logits = torch.randn(b, 1, h, w)
    mask_prob = torch.sigmoid(mask_logits)
    if mask_mode == "zero":
        mask_gt = torch.zeros(b, 1, h, w)
    elif mask_mode == "one":
        mask_gt = torch.ones(b, 1, h, w)
    else:
        mask_gt = torch.zeros(b, 1, h, w)
        mask_gt[:, :, :, : w // 2] = 1.0
    semantic_ir = torch.nn.functional.normalize(torch.randn(b, d, hf, wf), dim=1)
    semantic_vis = torch.nn.functional.normalize(torch.randn(b, d, hf, wf), dim=1)
    proto_attn = torch.softmax(torch.randn(b, hf * wf, k), dim=-1)
    x_ir_01 = torch.rand(b, 3, h, w)
    return {
        "pred_clear": pred_clear,
        "clear_gt": clear_gt,
        "density_map": density_map,
        "density_gt": density_gt,
        "mask_logits": mask_logits,
        "mask_prob": mask_prob,
        "mask_gt": mask_gt,
        "semantic_ir": semantic_ir,
        "semantic_vis": semantic_vis,
        "proto_attn": proto_attn,
        "x_ir_01": x_ir_01,
    }


def test_new_color_losses_are_skipped_when_weights_are_zero_without_intermediates():
    data = _base_tensors("mixed")
    loss = compute_teacher_region_loss(
        pred_clear=data["pred_clear"],
        clear_gt=data["clear_gt"],
        density_map=data["density_map"],
        density_gt=data["density_gt"],
        mask_logits=data["mask_logits"],
        mask_prob=data["mask_prob"],
        mask_gt=data["mask_gt"],
        lambda_rec=0.0,
        lambda_density=0.0,
        lambda_mask=0.0,
        lambda_align=0.0,
        lambda_comp=0.0,
        lambda_sparse=0.0,
        lambda_ir_tv=0.0,
    )

    for key in ("align", "comp", "sparse", "ir_tv"):
        assert key in loss
        assert loss[key].item() == 0.0


def test_new_color_losses_are_safe_for_zero_one_and_mixed_masks():
    for mode in ("zero", "one", "mixed"):
        data = _base_tensors(mode)
        loss = compute_teacher_region_loss(
            pred_clear=data["pred_clear"],
            clear_gt=data["clear_gt"],
            density_map=data["density_map"],
            density_gt=data["density_gt"],
            mask_logits=data["mask_logits"],
            mask_prob=data["mask_prob"],
            mask_gt=data["mask_gt"],
            semantic_ir=data["semantic_ir"],
            semantic_vis=data["semantic_vis"],
            proto_attn=data["proto_attn"],
            x_ir_01=data["x_ir_01"],
            lambda_rec=0.0,
            lambda_density=0.0,
            lambda_mask=0.0,
            lambda_align=0.1,
            lambda_comp=1.0,
            lambda_sparse=0.01,
            lambda_ir_tv=0.05,
        )

        for key in ("total", "align", "comp", "sparse", "ir_tv"):
            assert torch.isfinite(loss[key]), (mode, key)
        if mode == "one":
            assert loss["align"].item() == 0.0
        if mode == "zero":
            assert loss["comp"].item() == 0.0
            assert loss["sparse"].item() == 0.0
            assert loss["ir_tv"].item() == 0.0
