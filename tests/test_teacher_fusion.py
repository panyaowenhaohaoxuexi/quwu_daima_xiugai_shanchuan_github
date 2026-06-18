import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.teacher_fusion import BiDirectionalSemanticFusion
from model.teacher_semantic import SharedSemanticProjection
from model.Teacher import Bottle2neck, Res2Net, VIFNetInconsistencyTeacher


def _make_module_and_inputs(density_value=0.5, mask_value=0.0):
    torch.manual_seed(17)
    b, c, h, w = 2, 256, 4, 5
    shared_proj = SharedSemanticProjection(c, semantic_dim=16)
    module = BiDirectionalSemanticFusion(
        in_channels=c,
        semantic_dim=16,
        temperature=0.07,
        verify_threshold=0.2,
        verify_temperature=0.1,
        shared_proj=shared_proj,
    )
    fusion_head = nn.Sequential(nn.Conv2d(513, 1, 3, padding=1), nn.Sigmoid())
    F_vis = torch.randn(b, c, h, w)
    F_ir = torch.randn(b, c, h, w)
    density_map = torch.full((b, 1, h * 4, w * 4), float(density_value))
    mask = torch.full((b, 1, h, w), float(mask_value))
    return module, shared_proj, fusion_head, F_vis, F_ir, density_map, mask


def test_bidir_fusion_requires_shared_projection():
    with pytest.raises(ValueError, match="shared_proj"):
        BiDirectionalSemanticFusion(shared_proj=None)


def test_bidir_fusion_returns_expected_shapes_and_debug_fields():
    module, shared_proj, head, F_vis, F_ir, density, mask = _make_module_and_inputs()

    fused, debug = module(F_vis, F_ir, density, mask, head)

    assert module.shared_proj is shared_proj
    assert fused.shape == F_vis.shape
    assert debug["attn_v2i"].shape == (2, 20, 20)
    for key in ("verify_gate", "g", "fusion_score_map", "verify_score_map"):
        assert debug[key].shape == (2, 1, 4, 5)
    assert debug["inject"].shape == F_vis.shape
    for value in (fused, *debug.values()):
        assert torch.isfinite(value).all()


@pytest.mark.parametrize("density_value", [0.0, 1.0])
@pytest.mark.parametrize("mask_value", [0.0, 1.0])
def test_bidir_fusion_is_finite_at_density_and_mask_extremes(density_value, mask_value):
    module, _, head, F_vis, F_ir, density, mask = _make_module_and_inputs(
        density_value=density_value,
        mask_value=mask_value,
    )

    fused, debug = module(F_vis, F_ir, density, mask, head)

    assert torch.isfinite(fused).all()
    assert all(torch.isfinite(value).all() for value in debug.values())


def test_bidir_fusion_is_finite_for_mixed_binary_mask():
    module, _, head, F_vis, F_ir, density, mask = _make_module_and_inputs()
    mask[:, :, :, : mask.shape[-1] // 2] = 1.0

    fused, debug = module(F_vis, F_ir, density, mask, head)

    assert torch.isfinite(fused).all()
    assert all(torch.isfinite(value).all() for value in debug.values())


def test_bidir_fusion_uses_pure_ir_in_completion_region():
    module, _, head, F_vis, F_ir, density, mask = _make_module_and_inputs(mask_value=1.0)

    fused, _ = module(F_vis, F_ir, density, mask, head)

    assert torch.allclose(fused, F_ir, atol=1e-6)


def test_bidir_fusion_uses_asymmetric_residual_in_reliable_region():
    module, _, head, F_vis, F_ir, density, mask = _make_module_and_inputs(mask_value=0.0)

    fused, debug = module(F_vis, F_ir, density, mask, head)

    assert torch.allclose(fused, F_vis + debug["g"] * debug["inject"], atol=1e-6)


def test_fusion_score_map_is_raw_cosine_max_before_density_modulation():
    module, _, head, F_vis, F_ir, density, mask = _make_module_and_inputs(density_value=1.0)

    _, debug = module(F_vis, F_ir, density, mask, head)
    s_ir, s_vis = module.shared_proj(F_ir, F_vis)
    raw_v2i = torch.bmm(
        s_vis.flatten(2).transpose(1, 2),
        s_ir.flatten(2),
    )
    expected = raw_v2i.max(dim=-1).values.view(2, 1, 4, 5)

    assert torch.allclose(debug["fusion_score_map"], expected, atol=1e-6)


def _make_res2net_integration_inputs(mask_value):
    torch.manual_seed(23)
    encoder_ir = Res2Net(Bottle2neck, [1, 1, 1], baseWidth=26, scale=4).eval()
    encoder_vis = Res2Net(Bottle2neck, [1, 1, 1], baseWidth=26, scale=4).eval()
    x_ir = torch.randn(1, 3, 32, 32)
    x_vis = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        ir_feats, _ = encoder_ir(x_ir)
    shared_proj = SharedSemanticProjection(256, 16)
    bidir = BiDirectionalSemanticFusion(
        semantic_dim=16,
        shared_proj=shared_proj,
    )
    heads = nn.ModuleList([
        nn.Sequential(nn.Conv2d(2049, 1, 3, padding=1), nn.Sigmoid()),
        nn.Sequential(nn.Conv2d(1025, 1, 3, padding=1), nn.Sigmoid()),
        nn.Sequential(nn.Conv2d(513, 1, 3, padding=1), nn.Sigmoid()),
        nn.Sequential(nn.Conv2d(129, 1, 3, padding=1), nn.Sigmoid()),
    ])
    density = torch.rand(1, 1, 32, 32)
    mask = torch.full((1, 1, 32, 32), float(mask_value))
    return encoder_vis, x_vis, ir_feats, density, mask, heads, bidir


def test_res2net_all_completion_regions_use_unaligned_ir_at_every_scale():
    encoder, x_vis, ir_feats, density, mask, heads, bidir = _make_res2net_integration_inputs(1.0)

    with torch.no_grad():
        _, _, debug = encoder(
            x_vis,
            ir_feat_list=ir_feats,
            haze_mask=mask,
            density_map=density,
            fusion_weight_heads=heads,
            bidir_fusion=bidir,
            return_region_debug=True,
        )

    for fused, ir in zip(debug["fused_feats"], debug["ir_feats"]):
        assert torch.allclose(fused, ir, atol=1e-6)


def test_res2net_h4_reliable_region_matches_bidir_residual_and_keeps_debug_order():
    encoder, x_vis, ir_feats, density, mask, heads, bidir = _make_res2net_integration_inputs(0.0)

    with torch.no_grad():
        _, _, debug = encoder(
            x_vis,
            ir_feat_list=ir_feats,
            haze_mask=mask,
            density_map=density,
            fusion_weight_heads=heads,
            bidir_fusion=bidir,
            return_region_debug=True,
        )

    h4 = debug["fusion_debug_h4"]
    assert debug["vis_feats"][2].shape[1] == 256
    assert debug["ir_feats"][2].shape[1] == 256
    assert debug["fusion_weights"][2] is h4["g"]
    expected = debug["vis_feats"][2] + h4["g"] * h4["inject"]
    assert torch.allclose(debug["fused_feats"][2], expected, atol=1e-6)


def test_teacher_shares_one_semantic_projection_between_both_consumers():
    teacher = VIFNetInconsistencyTeacher(
        res_blocks=1,
        semantic_dim=16,
        num_color_prototypes=4,
    )

    assert teacher.color_transport.shared_proj is teacher.shared_semantic_proj
    assert teacher.bidir_fusion.shared_proj is teacher.shared_semantic_proj
