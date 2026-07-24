import os

import numpy as np
import pytest
import torch
from PIL import Image

from data.data_loader import (
    RealMultiModalDataset,
    collate_real,
    SynthMultiModalDataset,
    convert_density_semantics,
    load_scalar_map_as_float_tensor,
    load_tir_as_float_tensor,
)


def _write_rgb(path, value=128, size=(9, 7)):
    Image.new("RGB", size, (value, value, value)).save(path)


def test_scalar_16bit_transmission_preserves_adjacent_code_values(tmp_path):
    values = np.array([[1000, 1001]], dtype=np.uint16)
    path = tmp_path / "density.png"
    Image.fromarray(values, mode="I;16").save(path)

    raw = load_scalar_map_as_float_tensor(path)

    assert raw.shape == (1, 1, 2)
    assert raw.dtype == torch.float32
    assert raw[0, 0, 1] > raw[0, 0, 0]
    density = convert_density_semantics(raw, "transmission")
    assert density[0, 0, 1] < density[0, 0, 0]


def test_tir_three_channel_tolerance_uses_channel_mean(tmp_path):
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[..., 0] = 100
    image[..., 1] = 101
    image[..., 2] = 100
    path = tmp_path / "tir.png"
    Image.fromarray(image, mode="RGB").save(path)

    tir = load_tir_as_float_tensor(path)

    assert tir.shape == (3, 2, 3)
    assert torch.allclose(tir[0], tir[1])
    assert torch.allclose(tir[1], tir[2])
    assert tir[0, 0, 0] == pytest.approx(301 / 3 / 255)


def test_tir_three_channel_outside_tolerance_raises(tmp_path):
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[..., 0] = 10
    image[..., 1] = 40
    image[..., 2] = 10
    path = tmp_path / "tir.png"
    Image.fromarray(image, mode="RGB").save(path)

    with pytest.raises(ValueError, match="channel tolerance"):
        load_tir_as_float_tensor(path)


def test_synth_dataset_returns_four_items_without_completion_mask(tmp_path):
    for directory in ("clear", "ir", "hazy/mist", "Transmission_Map_GT/mist"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    _write_rgb(tmp_path / "clear" / "sample.png", value=20)
    _write_rgb(tmp_path / "ir" / "sample.png", value=30)
    _write_rgb(tmp_path / "hazy" / "mist" / "sample.png", value=40)
    Image.fromarray(np.full((7, 9), 65535, dtype=np.uint16), mode="I;16").save(
        tmp_path / "Transmission_Map_GT" / "mist" / "sample.png"
    )

    dataset = SynthMultiModalDataset(
        str(tmp_path), train=False, size=(7, 9), haze_levels=("mist",)
    )
    hazy, clear, tir, density = dataset[0]

    assert len((hazy, clear, tir, density)) == 4
    assert hazy.shape == clear.shape == tir.shape == (3, 7, 9)
    assert density.shape == (1, 7, 9)
    assert float(density.max()) == pytest.approx(0.0)
    assert float(hazy.min()) >= 0.0
    assert float(tir.max()) <= 1.0


def test_real_dataset_returns_only_hazy_tir_and_metadata(tmp_path):
    hazy_dir, tir_dir = tmp_path / "hazy", tmp_path / "tir"
    hazy_dir.mkdir()
    tir_dir.mkdir()
    _write_rgb(hazy_dir / "sample.png", value=40)
    Image.fromarray(np.full((7, 9), 1000, dtype=np.uint16), mode="I;16").save(tir_dir / "sample.png")

    dataset = RealMultiModalDataset(hazy_dir, tir_dir, pair_alignment_policy="strict")
    hazy, tir, metadata = dataset[0]

    assert hazy.shape == tir.shape == (3, 7, 9)
    assert set(metadata) >= {"sample_id", "filename", "original_size", "hazy_path", "tir_path"}
    assert "clear" not in metadata and "density" not in metadata and "mask" not in metadata


def test_real_collate_rejects_mixed_spatial_sizes_instead_of_silent_padding():
    first = (torch.zeros(3, 8, 8), torch.zeros(3, 8, 8), {"sample_id": "a"})
    second = (torch.zeros(3, 9, 8), torch.zeros(3, 9, 8), {"sample_id": "b"})

    with pytest.raises(ValueError, match="same spatial size"):
        collate_real([first, second])
