from pathlib import Path

import torch

from model.hde import HDE, IRDifferenceStructureEncoder


def _inputs(requires_grad=False):
    torch.manual_seed(31)
    x_vis = torch.rand(2, 3, 32, 32, requires_grad=requires_grad)
    x_ir = torch.rand(2, 3, 32, 32, requires_grad=requires_grad)
    return x_vis, x_ir


def test_ir_difference_encoder_uses_fixed_registered_kernels():
    encoder = IRDifferenceStructureEncoder()
    buffers = dict(encoder.named_buffers())
    parameters = dict(encoder.named_parameters())

    assert set(("cdc_kernel", "adc_kernels", "rdc_kernels")).issubset(buffers)
    assert buffers["cdc_kernel"].shape == (1, 1, 3, 3)
    assert buffers["adc_kernels"].shape == (2, 1, 3, 3)
    assert buffers["rdc_kernels"].shape == (2, 1, 3, 3)
    assert not any(name in parameters for name in buffers)

    expected_cdc = torch.tensor(
        [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]]
    ).view(1, 1, 3, 3)
    expected_adc = torch.tensor(
        [
            [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]],
            [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        ]
    ).unsqueeze(1)
    expected_rdc = torch.tensor(
        [
            [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ]
    ).unsqueeze(1)
    assert torch.equal(buffers["cdc_kernel"], expected_cdc)
    assert torch.equal(buffers["adc_kernels"], expected_adc)
    assert torch.equal(buffers["rdc_kernels"], expected_rdc)


def test_hde_dual_stream_shapes_range_and_density_feature_identity():
    hde = HDE().eval()
    x_vis, x_ir = _inputs()

    with torch.no_grad():
        density_map, density_feat, debug = hde(
            x_vis,
            x_ir,
            return_feat=True,
            return_debug=True,
        )

    assert density_map.shape == (2, 1, 32, 32)
    assert density_feat.shape == (2, 96, 32, 32)
    assert debug["ir_struct"].shape == (2, 32, 32, 32)
    assert debug["fm_vis"].shape == (2, 96, 32, 32)
    assert debug["fm_ir"].shape == (2, 96, 32, 32)
    assert hde.attn_conv.in_channels == 4
    assert torch.allclose(density_feat, debug["fm_vis"], atol=0, rtol=0)
    assert density_feat.data_ptr() == debug["fm_vis"].data_ptr()
    assert density_map.min() >= 0.0
    assert density_map.max() <= 1.0
    assert torch.isfinite(density_map).all()
    assert torch.isfinite(density_feat).all()


def test_hde_debug_fields_and_zero_initialized_offsets():
    hde = HDE().eval()
    x_vis, x_ir = _inputs()

    with torch.no_grad():
        _, _, debug = hde(x_vis, x_ir, return_feat=True, return_debug=True)

    expected_keys = {
        "ir_struct",
        "fm_vis",
        "fm_ir",
        "struct_diff_gap",
        "struct_diff_gmp",
        "offset1_vis",
        "offset1_ir",
        "offset2_vis",
        "offset2_ir",
    }
    assert expected_keys.issubset(debug)
    for key in ("offset1_vis", "offset1_ir", "offset2_vis", "offset2_ir"):
        assert torch.allclose(debug[key], torch.zeros_like(debug[key]), atol=1e-6)


def test_hde_supports_all_return_modes_and_legacy_ir_fallback():
    hde = HDE().eval()
    x_vis, x_ir = _inputs()

    with torch.no_grad():
        density_only = hde(x_vis)
        density_with_ir = hde(x_vis, x_ir)
        density_feat_pair = hde(x_vis, x_ir, return_feat=True)
        density_debug_pair = hde(x_vis, x_ir, return_debug=True)
        density_feat_debug = hde(x_vis, x_ir, return_feat=True, return_debug=True)

    assert density_only.shape == density_with_ir.shape == (2, 1, 32, 32)
    assert len(density_feat_pair) == 2
    assert len(density_debug_pair) == 2
    assert len(density_feat_debug) == 3


def test_hde_backward_reaches_visible_and_infrared_inputs():
    hde = HDE().eval()
    x_vis, x_ir = _inputs(requires_grad=True)

    density_map, density_feat, _ = hde(
        x_vis,
        x_ir,
        return_feat=True,
        return_debug=True,
    )
    (density_map.mean() + density_feat.mean()).backward()

    assert x_vis.grad is not None
    assert torch.isfinite(x_vis.grad).all()
    assert x_vis.grad.abs().sum() > 0
    assert x_ir.grad is not None
    assert torch.isfinite(x_ir.grad).all()
    assert x_ir.grad.abs().sum() > 0


def test_teacher_uses_normalized_ir_for_hde_and_keeps_mask_head_at_96_channels():
    root = Path(__file__).resolve().parents[1]
    source = (root / "model" / "Teacher.py").read_text(encoding="utf-8")

    assert "x_ir_01 = (x_ir * self.clip_input_std + self.clip_input_mean).clamp(0.0, 1.0)" in source
    assert "density_map, density_feat = self.hde(x_vis_01, x_ir_01, return_feat=True)" in source
    assert "nn.Conv2d(96, 32, kernel_size=3, padding=1)" in source
