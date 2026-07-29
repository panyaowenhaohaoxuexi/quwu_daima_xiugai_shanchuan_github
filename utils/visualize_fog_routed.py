"""V2 physical-mask routing diagnostics."""

import torch
from PIL import Image


def _to_rgb(value):
    value = value.detach().float().cpu()[0].clamp(0, 1)
    if value.shape[0] == 1:
        value = value.repeat(3, 1, 1)
    if value.shape[0] != 3:
        raise ValueError("visualized tensors must have one or three channels")
    return Image.fromarray(value.permute(1, 2, 0).mul(255).round().byte().numpy(), mode="RGB")


def build_diagnostic_panel(hazy_rgb, tir, clear_rgb, density_gt, completion_mask_gt, output):
    """Render hazy/TIR/clear/prediction, density, GT mask and predicted routes."""
    required = ("pred_clear", "density_map", "route_soft", "route_hard")
    missing = [key for key in required if key not in output]
    if missing:
        raise KeyError(f"formal output missing: {missing}")
    tiles = [_to_rgb(value) for value in (hazy_rgb, tir, clear_rgb, output["pred_clear"], density_gt,
             output["density_map"], completion_mask_gt, output["route_soft"], output["route_hard"])]
    width, height = tiles[0].size
    panel = Image.new("RGB", (width * len(tiles), height))
    for index, tile in enumerate(tiles):
        if tile.size != (width, height):
            raise ValueError("all diagnostic tensors must share spatial dimensions")
        panel.paste(tile, (index * width, 0))
    return panel
