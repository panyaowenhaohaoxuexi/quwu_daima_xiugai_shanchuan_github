import torch

from utils.visualize_fog_routed import build_diagnostic_panel


def test_formal_diagnostic_panel_uses_formal_fields_and_returns_rgb_image():
    sample = torch.rand(1, 3, 16, 17)
    scalar = torch.rand(1, 1, 16, 17)
    output = {
        "pred_clear": sample, "density_map": scalar, "route_soft": scalar,
        "route_hard": (scalar > 0.5).float(), "boundary_map": scalar,
    }
    panel = build_diagnostic_panel(sample, sample, sample, scalar, output, q=scalar, omega_support=scalar)

    assert panel.mode == "RGB"
    assert panel.size == (17 * 10, 16)
