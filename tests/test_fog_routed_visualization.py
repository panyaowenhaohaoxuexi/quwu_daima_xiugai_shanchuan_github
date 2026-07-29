import torch

from utils.visualize_fog_routed import build_diagnostic_panel


def test_v2_diagnostic_panel_has_the_nine_physical_mask_route_tiles():
    rgb = torch.zeros(1, 3, 4, 5)
    scalar = torch.zeros(1, 1, 4, 5)
    output = {"pred_clear": rgb, "density_map": scalar, "route_soft": scalar, "route_hard": scalar}
    panel = build_diagnostic_panel(rgb, rgb, rgb, scalar, scalar, output)
    assert panel.size == (5 * 9, 4)
