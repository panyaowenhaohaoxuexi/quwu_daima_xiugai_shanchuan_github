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

    for key in ("align", "comp", "comp_perc", "sparse", "ir_tv"):
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
            lambda_comp_perc=0.5,
            lambda_sparse=0.01,
            lambda_ir_tv=0.05,
        )

        for key in ("total", "align", "comp", "comp_perc", "sparse", "ir_tv"):
            assert torch.isfinite(loss[key]), (mode, key)
        if mode == "one":
            assert loss["align"].item() == 0.0
        if mode == "zero":
            assert loss["comp"].item() == 0.0
            assert loss["comp_perc"].item() == 0.0
            assert loss["sparse"].item() == 0.0
            assert loss["ir_tv"].item() == 0.0


def test_infonce_align_is_finite_for_empty_single_and_repeated_semantics():
    for mode in ("one", "mixed"):
        data = _base_tensors(mode)
        if mode == "mixed":
            data["mask_gt"].fill_(1.0)
            data["mask_gt"][:, :, 0, 0] = 0.0
        data["semantic_vis"][:, :, 0, 0] = data["semantic_vis"][:, :, 0, 1]
        data["semantic_vis"] = torch.nn.functional.normalize(data["semantic_vis"], dim=1)
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
            lambda_rec=0.0,
            lambda_density=0.0,
            lambda_mask=0.0,
            lambda_align=0.1,
            align_mode="infonce",
        )

        assert torch.isfinite(loss["align"]), mode
        assert loss["align"].item() >= 0.0
        assert loss["align"].item() == 0.0


def test_infonce_align_is_symmetric_when_swapping_modalities():
    from loss.teacher_region_loss import _info_nce_align

    data = _base_tensors("zero")
    mask = torch.ones(2, 1, 4, 4)
    loss_ir_vis = _info_nce_align(
        data["semantic_ir"],
        data["semantic_vis"],
        mask,
        temperature=0.07,
        fp_threshold=0.8,
        max_samples=1024,
    )
    loss_vis_ir = _info_nce_align(
        data["semantic_vis"],
        data["semantic_ir"],
        mask,
        temperature=0.07,
        fp_threshold=0.8,
        max_samples=1024,
    )

    assert torch.isfinite(loss_ir_vis)
    assert torch.allclose(loss_ir_vis, loss_vis_ir, atol=1e-6)


def test_infonce_false_positive_threshold_controls_masking():
    from loss.teacher_region_loss import _info_nce_align

    semantic_ir = torch.nn.functional.normalize(torch.eye(4).view(1, 4, 2, 2), dim=1)
    semantic_vis = semantic_ir.clone()
    reliable_mask = torch.ones(1, 1, 2, 2)

    loss_no_mask, stats_no_mask = _info_nce_align(
        semantic_ir,
        semantic_vis,
        reliable_mask,
        temperature=0.07,
        fp_threshold=1.0,
        max_samples=1024,
        return_stats=True,
    )
    loss_with_mask, stats_with_mask = _info_nce_align(
        semantic_ir,
        semantic_vis,
        reliable_mask,
        temperature=0.07,
        fp_threshold=-0.1,
        max_samples=1024,
        return_stats=True,
    )

    assert torch.isfinite(loss_no_mask)
    assert torch.isfinite(loss_with_mask)
    assert stats_no_mask["masked_i2v"] == 0
    assert stats_no_mask["masked_v2i"] == 0
    assert stats_with_mask["masked_i2v"] > 0
    assert stats_with_mask["masked_v2i"] > 0


def test_infonce_sampling_caps_actual_sample_count_and_keeps_gradients():
    from loss.teacher_region_loss import _info_nce_align

    torch.manual_seed(13)
    semantic_ir = torch.nn.functional.normalize(torch.randn(1, 8, 8, 8), dim=1).requires_grad_()
    semantic_vis = torch.nn.functional.normalize(torch.randn(1, 8, 8, 8), dim=1).requires_grad_()
    reliable_mask = torch.ones(1, 1, 8, 8)

    losses = []
    for _ in range(3):
        loss, stats = _info_nce_align(
            semantic_ir,
            semantic_vis,
            reliable_mask,
            temperature=0.07,
            fp_threshold=1.0,
            max_samples=5,
            return_stats=True,
        )
        assert torch.isfinite(loss)
        assert stats["sample_counts"] == [5]
        losses.append(loss)

    losses[-1].backward()
    assert semantic_ir.grad is not None
    assert semantic_vis.grad is not None
    assert torch.isfinite(semantic_ir.grad).all()
    assert torch.isfinite(semantic_vis.grad).all()
