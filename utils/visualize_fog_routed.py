"""Formal fog-routed diagnostic visualization without legacy fields."""

from __future__ import annotations

import torch
from PIL import Image


def _to_rgb(value):
    value = value.detach().float().cpu()[0].clamp(0, 1)
    if value.shape[0] == 1:
        value = value.repeat(3, 1, 1)
    if value.shape[0] != 3:
        raise ValueError("visualized tensors must have one or three channels")
    array = value.permute(1, 2, 0).mul(255).round().byte().numpy()
    return Image.fromarray(array, mode="RGB")


def build_diagnostic_panel(hazy_rgb, tir, clear_rgb, density_gt, output, *, q=None, omega_support=None):
    """Return one horizontal RGB panel using only the formal model interface."""
    required = ("pred_clear", "density_map", "route_soft", "route_hard", "boundary_map")
    missing = [key for key in required if key not in output]
    if missing:
        raise KeyError(f"formal output missing: {missing}")
    q_or_omega = q if q is not None else omega_support
    if q_or_omega is None:
        q_or_omega = torch.zeros_like(output["boundary_map"])
    tiles = [
        _to_rgb(hazy_rgb), _to_rgb(tir), _to_rgb(clear_rgb), _to_rgb(output["pred_clear"]),
        _to_rgb(density_gt), _to_rgb(output["density_map"]), _to_rgb(output["route_soft"]),
        _to_rgb(output["route_hard"]), _to_rgb(output["boundary_map"]), _to_rgb(q_or_omega),
    ]
    width, height = tiles[0].size
    panel = Image.new("RGB", (width * len(tiles), height))
    for index, tile in enumerate(tiles):
        if tile.size != (width, height):
            raise ValueError("all diagnostic tensors must share spatial dimensions")
        panel.paste(tile, (index * width, 0))
    return panel
