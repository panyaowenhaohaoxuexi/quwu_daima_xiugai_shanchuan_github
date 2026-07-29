import numpy as np
import subprocess
import sys
from pathlib import Path

from PIL import Image

from tools.generate_auto_route_gt import build_route_gt, generate_split


def test_route_gt_marks_only_opaque_tir_structure():
    density = np.full((17, 17), 0.99, dtype=np.float32)
    density[:, :5] = 0.20
    tir = np.zeros((17, 17), dtype=np.float32)
    tir[:, 8:] = 1.0

    route = build_route_gt(density, tir, transmission_max=0.05)

    assert route.dtype == np.uint8
    assert route.shape == density.shape
    assert route[:, :5].max() == 0
    assert route[:, 7:10].max() == 255


def test_route_gt_rejects_opaque_but_structureless_region():
    density = np.full((17, 17), 0.99, dtype=np.float32)
    tir = np.full((17, 17), 0.5, dtype=np.float32)

    route = build_route_gt(density, tir, transmission_max=0.05)

    assert not route.any()


def test_route_gt_rejects_structured_low_fog_region():
    density = np.full((17, 17), 0.20, dtype=np.float32)
    tir = np.zeros((17, 17), dtype=np.float32)
    tir[:, 8:] = 1.0

    route = build_route_gt(density, tir, transmission_max=0.05)

    assert not route.any()


def test_generator_cli_can_import_project_modules_from_tools_directory():
    root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "tools/generate_auto_route_gt.py", "--help"],
        cwd=root, capture_output=True, text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--data-root" in result.stdout


def test_generate_split_can_resume_in_bounded_batches(tmp_path):
    for directory in ("train/ir", "train/hazy/local_extreme", "train/Transmission_Map_GT/local_extreme"):
        (tmp_path / directory).mkdir(parents=True)
    density = np.full((9, 9), 65535, dtype=np.uint16)
    tir = np.zeros((9, 9), dtype=np.uint8)
    tir[:, 4:] = 255
    for stem in ("a", "b"):
        Image.fromarray(np.dstack((tir, tir, tir)), mode="RGB").save(tmp_path / "train/ir" / f"{stem}.png")
        Image.fromarray(np.dstack((tir, tir, tir)), mode="RGB").save(tmp_path / "train/hazy/local_extreme" / f"{stem}.png")
        Image.fromarray(density, mode="I;16").save(tmp_path / "train/Transmission_Map_GT/local_extreme" / f"{stem}.png")

    first = generate_split(tmp_path, "train", ("local_extreme",), max_items=1)
    second = generate_split(tmp_path, "train", ("local_extreme",), max_items=1)

    assert first["local_extreme"]["written"] == 1
    assert second["local_extreme"]["written"] == 1
    assert len(list((tmp_path / "train/Auto_Route_GT/local_extreme").glob("*.png"))) == 2
