"""Paper-oriented rendering for completion-stream provenance visualizations."""

from __future__ import annotations

from typing import Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch


_NAVY = "#172033"
_SLATE = "#475569"
_RED = "#ef4444"
_TEAL = "#0f766e"
_PURPLE = "#7c3aed"


def select_structured_completion_query(
    route: torch.Tensor,
    structure_energy: torch.Tensor,
    border_fraction: float = 0.12,
) -> tuple[int, int]:
    """Select a completion query that is both routed to completion and structurally informative."""
    if route.shape != structure_energy.shape or route.ndim != 4 or route.shape[1] != 1:
        raise ValueError("route and structure_energy must have identical [B, 1, H, W] shapes")
    if not 0.0 <= border_fraction < 0.5:
        raise ValueError("border_fraction must lie in [0, 0.5)")
    route_map = route[0, 0].float()
    energy = structure_energy[0, 0].float()
    energy = (energy - energy.amin()) / (energy.amax() - energy.amin()).clamp_min(1e-8)
    height, width = route_map.shape
    border_h = int(round(height * border_fraction))
    border_w = int(round(width * border_fraction))
    interior = torch.ones_like(route_map, dtype=torch.bool)
    if border_h > 0 and height > 2 * border_h:
        interior[:border_h] = False
        interior[-border_h:] = False
    if border_w > 0 and width > 2 * border_w:
        interior[:, :border_w] = False
        interior[:, -border_w:] = False
    completion = (route_map >= 0.5) & interior
    if not completion.any():
        completion = route_map >= 0.5
    score = route_map * energy
    if completion.any():
        score = score.masked_fill(~completion, float("-inf"))
    index = int(score.reshape(-1).argmax().item())
    return index // route_map.shape[1], index % route_map.shape[1]


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (["arialbd.ttf", "Arial Bold.ttf"] if bold else ["arial.ttf", "Arial.ttf"]) + ["simhei.ttf"]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _fit(canvas: Image.Image, source: Image.Image, box: tuple[int, int, int, int]) -> tuple[int, int, float]:
    x, y, width, height = box
    image = source.convert("RGB")
    resized = ImageOps.contain(image, (width, height), Image.Resampling.LANCZOS)
    origin = (x + (width - resized.width) // 2, y + (height - resized.height) // 2)
    canvas.paste(resized, origin)
    return origin[0], origin[1], resized.width / image.width


def _draw_box(draw: ImageDraw.ImageDraw, xy: tuple[int, int], half: int, color: str, width: int = 4) -> None:
    x, y = xy
    draw.rectangle((x - half, y - half, x + half, y + half), outline=color, width=width)


def compose_completion_panel(
    *,
    hazy: Image.Image,
    infrared: Image.Image,
    attention: Image.Image,
    output: Image.Image,
    patches: Sequence[Image.Image],
    weights: Sequence[float],
    query_xy: tuple[int, int],
    source_xy: Sequence[tuple[int, int]] | None = None,
) -> Image.Image:
    """Compose a single completion-stream provenance figure from real model outputs.

    ``query_xy`` and ``source_xy`` use original-image (x, y) coordinates.
    """
    if len(patches) != 3 or len(weights) != 3:
        raise ValueError("exactly three reference patches and weights are required")

    panel = Image.new("RGB", (1560, 540), "white")
    draw = ImageDraw.Draw(panel)
    draw.text((42, 28), "Completion stream: IR-structure-guided appearance retrieval", fill=_NAVY, font=_font(28, True))
    draw.text((42, 64), "IR preserves the missing geometry; appearance is retrieved from reliable low-routing RGB regions.", fill=_SLATE, font=_font(17))

    image_top, image_height, image_width = 135, 270, 300
    boxes = {
        "hazy": (40, image_top, image_width, image_height),
        "infrared": (400, image_top, image_width, image_height),
        "attention": (760, image_top, image_width, image_height),
        "output": (1230, image_top, image_width, image_height),
    }
    labels = {
        "hazy": "(a) Dense hazy RGB query",
        "infrared": "(b) IR structural query",
        "attention": "(c) Top-1 reliable RGB appearance source",
        "output": "(d) Completed output",
    }
    for key, (x, _, _, _) in boxes.items():
        draw.text((x, 106), labels[key], fill=_NAVY, font=_font(18, True))

    hx, hy, hscale = _fit(panel, hazy, boxes["hazy"])
    ix, iy, iscale = _fit(panel, infrared, boxes["infrared"])
    ax, ay, ascale = _fit(panel, attention, boxes["attention"])
    ox, oy, oscale = _fit(panel, output, boxes["output"])
    query_x, query_y = query_xy
    for x0, y0, scale, color in ((hx, hy, hscale, _RED), (ix, iy, iscale, _RED), (ox, oy, oscale, _PURPLE)):
        _draw_box(draw, (round(x0 + query_x * scale), round(y0 + query_y * scale)), max(18, round(36 * scale)), color)

    if source_xy:
        source_x, source_y = source_xy[0]
        _draw_box(
            draw,
            (round(ax + source_x * ascale), round(ay + source_y * ascale)),
            max(14, round(30 * ascale)),
            "#14b8a6",
            4,
        )
    inset_x, inset_y, inset_size = 960, 292, (120, 96)
    inset = ImageOps.fit(patches[0].convert("RGB"), inset_size, Image.Resampling.LANCZOS)
    panel.paste(inset, (inset_x, inset_y))
    draw.rectangle((inset_x, inset_y, inset_x + inset_size[0] - 1, inset_y + inset_size[1] - 1), outline="#14b8a6", width=4)
    draw.text((930, 414), f"Top-1 source, w={weights[0]:.3f}", fill=_TEAL, font=_font(15, True))

    arrow_y = image_top + image_height // 2
    draw.line((1085, arrow_y, 1174, arrow_y), fill=_SLATE, width=5)
    draw.polygon([(1174, arrow_y - 10), (1200, arrow_y), (1174, arrow_y + 10)], fill=_SLATE)
    draw.text((1084, arrow_y - 34), "RGB appearance", fill=_TEAL, font=_font(15, True))
    draw.line((704, 205, 1174, 205), fill="#8b5cf6", width=3)
    draw.polygon([(1174, 197), (1200, 205), (1174, 213)], fill="#8b5cf6")
    draw.text((910, 176), "IR geometry", fill="#7c3aed", font=_font(15, True))

    draw.rounded_rectangle((40, 465, 1520, 510), radius=8, fill="#eef2ff", outline="#c7d2fe", width=2)
    draw.text((58, 479), "Completed output = IR-consistent structure + retrieved RGB appearance (not direct IR-to-color translation).", fill=_NAVY, font=_font(15))
    return panel
