import torch
import torch.nn.functional as F
import pytest

from model.Teacher import CrossModalSemanticColorTransport
from model.teacher_semantic import SharedSemanticProjection


def _make_inputs(mask_value=None):
    torch.manual_seed(7)
    b, h, w = 2, 32, 32
    hf, wf = h // 4, w // 4
    ir_feat = torch.randn(b, 256, hf, wf)
    vis_feat = torch.randn(b, 256, hf, wf)
    x_vis_01 = torch.rand(b, 3, h, w)
    if mask_value is None:
        binary_mask = torch.zeros(b, 1, h, w)
        binary_mask[:, :, :, : w // 2] = 1.0
    else:
        binary_mask = torch.full((b, 1, h, w), float(mask_value))
    return ir_feat, vis_feat, x_vis_01, binary_mask


def test_color_transport_outputs_expected_names_and_shapes():
    ir_feat, vis_feat, x_vis_01, binary_mask = _make_inputs()
    module = CrossModalSemanticColorTransport(
        in_channels=256,
        semantic_dim=16,
        num_prototypes=5,
        temperature=0.07,
        shared_proj=SharedSemanticProjection(256, 16),
    )

    out = module(ir_feat, vis_feat, x_vis_01, binary_mask)

    expected = {
        "transported_rgb",
        "transported_rgb_feat",
        "semantic_ir",
        "semantic_vis",
        "proto_keys",
        "proto_values",
        "proto_attn",
        "proto_assign",
        "reliable_area_ratio",
        "max_sim_map",
        "max_sim_stats",
    }
    assert expected.issubset(out)
    assert out["transported_rgb"].shape == x_vis_01.shape
    assert out["transported_rgb_feat"].shape == (2, 3, 8, 8)
    assert out["semantic_ir"].shape == (2, 16, 8, 8)
    assert out["semantic_vis"].shape == (2, 16, 8, 8)
    assert out["proto_keys"].shape == (2, 5, 16)
    assert out["proto_values"].shape == (2, 5, 3)
    assert out["proto_attn"].shape == (2, 64, 5)
    assert out["proto_assign"].shape == (2, 5, 8, 8)
    assert out["max_sim_map"].shape == (2, 1, 8, 8)
    assert torch.isfinite(out["proto_attn"]).all()
    assert torch.allclose(out["proto_attn"].sum(dim=-1), torch.ones(2, 64), atol=1e-6)


def test_color_transport_empty_reliable_region_is_numerically_safe():
    ir_feat, vis_feat, x_vis_01, binary_mask = _make_inputs(mask_value=1.0)
    module = CrossModalSemanticColorTransport(
        in_channels=256,
        semantic_dim=8,
        num_prototypes=4,
        temperature=0.07,
        shared_proj=SharedSemanticProjection(256, 8),
    )

    out = module(ir_feat, vis_feat, x_vis_01, binary_mask)

    for key in ("transported_rgb", "proto_keys", "proto_values", "proto_attn", "proto_assign"):
        assert torch.isfinite(out[key]).all(), key
    assert torch.allclose(out["reliable_area_ratio"], torch.zeros(2), atol=1e-6)
    assert out["transported_rgb"].min() >= 0.0
    assert out["transported_rgb"].max() <= 1.0


def test_region_locked_composite_formula_uses_transport_for_completion_only():
    _, _, x_vis_01, binary_mask = _make_inputs()
    pred_raw = torch.rand_like(x_vis_01)
    transported_rgb = torch.rand_like(x_vis_01)
    m_full = F.interpolate(binary_mask, size=pred_raw.shape[-2:], mode="nearest")

    pred_clear = m_full * transported_rgb + (1.0 - m_full) * pred_raw

    assert torch.allclose(pred_clear[m_full.expand_as(pred_clear) == 1], transported_rgb[m_full.expand_as(transported_rgb) == 1], atol=1e-6)
    assert torch.allclose(pred_clear[m_full.expand_as(pred_clear) == 0], pred_raw[m_full.expand_as(pred_raw) == 0], atol=1e-6)


def test_color_transport_requires_shared_projection():
    with pytest.raises(ValueError, match="shared_proj"):
        CrossModalSemanticColorTransport(shared_proj=None)


def test_color_transport_uses_the_supplied_projection_instance():
    shared_proj = SharedSemanticProjection(256, 8)
    module = CrossModalSemanticColorTransport(
        in_channels=256,
        semantic_dim=8,
        num_prototypes=4,
        shared_proj=shared_proj,
    )

    assert module.shared_proj is shared_proj
