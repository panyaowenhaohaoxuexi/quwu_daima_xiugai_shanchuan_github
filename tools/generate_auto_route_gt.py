"""Generate automatic binary completion-route labels from density GT and TIR.

The generated label is training-only: 255 denotes a physically opaque RGB region
that still contains local TIR structure (completion); 0 denotes fusion.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_loader import (
    SYNTH_IMAGE_EXTS,
    load_scalar_map_as_float_tensor,
    load_tir_as_float_tensor,
)


DEFAULT_SPLITS = ("train", "test")
DEFAULT_HAZE_LEVELS = ("mist", "middle", "dense", "local_extreme")


def _require_unit_scalar(name: str, value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [H,W], received {value.shape}")
    if not value.size or not np.isfinite(value).all():
        raise ValueError(f"{name} must be non-empty and finite")
    if float(value.min()) < 0.0 or float(value.max()) > 1.0:
        raise ValueError(f"{name} must lie in [0,1], range=({float(value.min())}, {float(value.max())})")
    return value


def tir_structure_mask(tir: np.ndarray) -> np.ndarray:
    """Return an image-adaptive binary mask of local TIR contours.

    Otsu chooses the edge boundary separately for every image, avoiding one
    manually fixed global edge magnitude. A one-pixel dilation turns contours
    into local regions suitable for routing supervision.
    """
    tir = _require_unit_scalar("tir", tir)
    gx = cv2.Sobel(tir, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(tir, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    maximum = float(magnitude.max())
    if maximum <= 1e-8:
        return np.zeros(tir.shape, dtype=bool)
    scaled = np.uint8(np.clip(magnitude / maximum, 0.0, 1.0) * 255.0)
    _, thresholded = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.dilate(thresholded, np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)


def build_route_gt(density_m: np.ndarray, tir: np.ndarray, *, transmission_max: float = 0.05) -> np.ndarray:
    """Return `uint8` route GT: 255=hard completion and 0=fusion.

    `transmission_max` is only the physical data-generation definition of an
    almost opaque direct-RGB component (`T=1-M`), never an inference-time
    density threshold for the learned router.
    """
    if not 0.0 < float(transmission_max) < 1.0:
        raise ValueError("transmission_max must lie in (0,1)")
    density_m = _require_unit_scalar("density_m", density_m)
    tir = _require_unit_scalar("tir", tir)
    if density_m.shape != tir.shape:
        raise ValueError(f"density_m and tir must have matching shapes: {density_m.shape} != {tir.shape}")
    opaque = (1.0 - density_m) <= float(transmission_max)
    return np.where(opaque & tir_structure_mask(tir), 255, 0).astype(np.uint8)


def _image_index(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(f"missing image directory: {directory}")
    index: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SYNTH_IMAGE_EXTS:
            continue
        stem = path.stem.lower()
        if stem in index:
            raise ValueError(f"duplicate stem {stem!r} in {directory}")
        index[stem] = path
    return index


def _read_density_m(path: Path) -> np.ndarray:
    return load_scalar_map_as_float_tensor(path).squeeze(0).numpy()


def _read_tir(path: Path) -> np.ndarray:
    return load_tir_as_float_tensor(path).mean(dim=0).numpy()


def generate_split(
    data_root: Path,
    split: str,
    haze_levels: Iterable[str],
    *,
    output_dirname: str = "Auto_Route_GT",
    transmission_max: float = 0.05,
    overwrite: bool = False,
    max_items: int | None = None,
) -> dict[str, dict[str, float]]:
    """Generate labels for one FLIR split and return per-level summaries."""
    split_root = data_root / split
    if max_items is not None and int(max_items) < 1:
        raise ValueError("max_items must be positive when supplied")
    tir_index = _image_index(split_root / "ir")
    summaries: dict[str, dict[str, float]] = {}
    for level in haze_levels:
        density_index = _image_index(split_root / "Transmission_Map_GT" / level)
        hazy_index = _image_index(split_root / "hazy" / level)
        missing_density = sorted(set(hazy_index) - set(density_index))
        missing_tir = sorted(set(hazy_index) - set(tir_index))
        if missing_density or missing_tir:
            raise FileNotFoundError(
                f"unpaired inputs split={split} level={level}: "
                f"missing_density={missing_density[:3]} missing_tir={missing_tir[:3]}"
            )
        output_dir = split_root / output_dirname / level
        output_dir.mkdir(parents=True, exist_ok=True)
        written, skipped, positive_pixels, total_pixels = 0, 0, 0, 0
        for stem in sorted(hazy_index):
            if max_items is not None and written >= int(max_items):
                break
            output_path = output_dir / f"{hazy_index[stem].stem}.png"
            if output_path.exists() and not overwrite:
                skipped += 1
                continue
            density_m, tir = _read_density_m(density_index[stem]), _read_tir(tir_index[stem])
            if density_m.shape != tir.shape:
                raise ValueError(
                    f"TIR/density geometry mismatch: density={density_index[stem]} shape={density_m.shape}, "
                    f"tir={tir_index[stem]} shape={tir.shape}"
                )
            route = build_route_gt(density_m, tir, transmission_max=transmission_max)
            Image.fromarray(route, mode="L").save(output_path)
            written += 1
            positive_pixels += int((route > 0).sum())
            total_pixels += int(route.size)
        summaries[level] = {
            "hazy_count": float(len(hazy_index)), "written": float(written), "skipped": float(skipped),
            "positive_fraction_written": float(positive_pixels / total_pixels) if total_pixels else 0.0,
        }
    return summaries


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--splits", nargs="+", default=DEFAULT_SPLITS, choices=DEFAULT_SPLITS)
    parser.add_argument("--haze-levels", nargs="+", default=DEFAULT_HAZE_LEVELS)
    parser.add_argument("--output-dirname", default="Auto_Route_GT")
    parser.add_argument("--transmission-max", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-items", type=int, default=None,
                        help="Write at most this many missing labels per split/level; safe for resumable batches.")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {data_root}")
    for split in args.splits:
        summary = generate_split(
            data_root, split, args.haze_levels, output_dirname=args.output_dirname,
            transmission_max=args.transmission_max, overwrite=args.overwrite, max_items=args.max_items,
        )
        for level, values in summary.items():
            print(
                f"split={split} level={level} hazy={int(values['hazy_count'])} "
                f"written={int(values['written'])} skipped={int(values['skipped'])} "
                f"positive_fraction_written={values['positive_fraction_written']:.6f}"
            )


if __name__ == "__main__":
    main()
