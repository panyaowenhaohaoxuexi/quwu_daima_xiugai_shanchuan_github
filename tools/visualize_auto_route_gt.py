"""Create compact visual audits for automatically generated route labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import SYNTH_IMAGE_EXTS, load_scalar_map_as_float_tensor, load_tir_as_float_tensor


def _index(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"missing image directory: {directory}")
    return {path.stem.lower(): path for path in sorted(directory.iterdir()) if path.suffix.lower() in SYNTH_IMAGE_EXTS}


def _gray_image(value: np.ndarray) -> Image.Image:
    array = np.uint8(np.clip(value, 0, 1) * 255)
    return Image.fromarray(array, mode="L").convert("RGB")


def build_panel(hazy: Image.Image, tir: np.ndarray, density: np.ndarray, route: np.ndarray) -> Image.Image:
    """Return a 2x2 panel: hazy RGB, TIR, density M, and route overlay."""
    hazy = hazy.convert("RGB")
    width, height = hazy.size
    overlay = np.asarray(hazy).copy()
    mask = route > 0
    overlay[mask] = np.array([255, 0, 0], dtype=np.uint8)
    tiles = (hazy, _gray_image(tir), _gray_image(density), Image.fromarray(overlay, mode="RGB"))
    labels = ("Hazy RGB", "TIR", "Fog density M", "Auto route GT (red=completion)")
    panel = Image.new("RGB", (width * 2, (height + 24) * 2), "white")
    draw = ImageDraw.Draw(panel)
    for index, (tile, label) in enumerate(zip(tiles, labels)):
        x, y = (index % 2) * width, (index // 2) * (height + 24)
        panel.paste(tile.resize((width, height)), (x, y))
        draw.text((x + 4, y + height + 4), label, fill="black")
    return panel


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--haze-level", required=True)
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    root = args.data_root / args.split
    hazy = _index(root / "hazy" / args.haze_level)
    density = _index(root / "Transmission_Map_GT" / args.haze_level)
    route = _index(root / "Auto_Route_GT" / args.haze_level)
    tir = _index(root / "ir")
    stems = sorted(set(hazy) & set(density) & set(route) & set(tir))
    if not stems:
        raise ValueError("no complete hazy/TIR/density/route quartets found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems[:args.count]:
        with Image.open(hazy[stem]) as image:
            hazy_image = image.copy()
        tir_array = load_tir_as_float_tensor(tir[stem]).mean(dim=0).numpy()
        density_array = load_scalar_map_as_float_tensor(density[stem]).squeeze(0).numpy()
        with Image.open(route[stem]) as image:
            route_array = np.asarray(image.convert("L"))
        build_panel(hazy_image, tir_array, density_array, route_array).save(args.output_dir / f"{hazy[stem].stem}_route_gt.png")
    print(f"Saved {min(args.count, len(stems))} route-GT panels to {args.output_dir}")


if __name__ == "__main__":
    main()
