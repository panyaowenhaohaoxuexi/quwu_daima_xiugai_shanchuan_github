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


def test_tir_per_image_percentile_normalizes_after_raw_channel_mean(tmp_path):
    image = np.array([[[10, 11, 10], [20, 21, 20]], [[30, 31, 30], [40, 41, 40]]], dtype=np.uint8)
    path = tmp_path / "tir.png"
    Image.fromarray(image, mode="RGB").save(path)

    tir = load_tir_as_float_tensor(path, {
        "normalization": "percentile", "percentile_scope": "per_image",
        "percentile_low": 0.0, "percentile_high": 100.0,
        "channel_tolerance_code_values": 1,
    })

    assert tir[0, 0, 0] == pytest.approx(0.0)
    assert tir[0, 1, 1] == pytest.approx(1.0)
    assert torch.equal(tir[0], tir[1]) and torch.equal(tir[1], tir[2])


def test_tir_dataset_percentile_uses_fixed_calibrated_range(tmp_path):
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    Image.fromarray(np.array([[10, 20]], dtype=np.uint16), mode="I;16").save(first)
    Image.fromarray(np.array([[20, 30]], dtype=np.uint16), mode="I;16").save(second)
    config = {
        "normalization": "percentile", "percentile_scope": "dataset",
        "dataset_percentile_low_value": 10.0, "dataset_percentile_high_value": 30.0,
    }

    assert load_tir_as_float_tensor(first, config)[0, 0, 1] == pytest.approx(0.5)
    assert load_tir_as_float_tensor(second, config)[0, 0, 0] == pytest.approx(0.5)


def test_tir_constant_per_image_percentile_is_finite_zero(tmp_path):
    path = tmp_path / "tir.png"
    Image.fromarray(np.full((2, 2), 42, dtype=np.uint16), mode="I;16").save(path)

    tir = load_tir_as_float_tensor(path, {
        "normalization": "percentile", "percentile_scope": "per_image",
        "percentile_low": 10.0, "percentile_high": 90.0,
    })

    assert torch.isfinite(tir).all()
    assert torch.equal(tir, torch.zeros_like(tir))


@pytest.mark.parametrize("invalid", [np.array([], dtype=np.float32), np.array([[np.nan]], dtype=np.float32), np.array([[np.inf]], dtype=np.float32)])
def test_tir_every_normalization_path_rejects_empty_and_nonfinite_values(monkeypatch, invalid):
    import data.data_loader as loader

    monkeypatch.setattr(loader, "_open_preserving_known_bit_depth", lambda _path: ("F", invalid, None))
    for config in ({"normalization": "dtype_range"}, {"normalization": "fixed_range", "fixed_min": 0, "fixed_max": 1},
                   {"normalization": "percentile", "percentile_scope": "per_image"}):
        with pytest.raises(ValueError, match="invalid-tir"):
            loader.load_tir_as_float_tensor("invalid-tir", config)


def test_signed_16_container_rejects_values_outside_unsigned_range():
    import data.data_loader as loader

    with pytest.raises(ValueError, match="invalid unsigned 16-bit range in signed-16"):
        loader._array_to_unit_float(np.array([[-1, 3]], dtype=np.int32), path="signed-16", known_integer_bits=16)
    with pytest.raises(ValueError, match="max=70000"):
        loader._array_to_unit_float(np.array([[70000]], dtype=np.int32), path="signed-16", known_integer_bits=16)


def test_synth_dataset_pairs_tiff_tir_and_density_by_stem(tmp_path):
    for directory in ("clear", "ir", "hazy/mist", "Transmission_Map_GT/mist"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    _write_rgb(tmp_path / "clear" / "sample.png", value=20)
    _write_rgb(tmp_path / "hazy" / "mist" / "sample.png", value=40)
    Image.fromarray(np.full((7, 9), 1000, dtype=np.uint16), mode="I;16").save(tmp_path / "ir" / "sample.tiff")
    Image.fromarray(np.full((7, 9), 50000, dtype=np.uint16), mode="I;16").save(
        tmp_path / "Transmission_Map_GT" / "mist" / "sample.tif"
    )

    dataset = SynthMultiModalDataset(str(tmp_path), train=False, haze_levels=("mist",))
    assert len(dataset) == 1
    assert dataset[0][2].shape == (3, 7, 9)


@pytest.mark.parametrize("directory", ["clear", "ir", "Transmission_Map_GT/mist", "hazy/mist"])
def test_synth_dataset_rejects_duplicate_stems_within_one_directory(tmp_path, directory):
    for parent in ("clear", "ir", "hazy/mist", "Transmission_Map_GT/mist"):
        (tmp_path / parent).mkdir(parents=True, exist_ok=True)
    _write_rgb(tmp_path / "clear" / "sample.png", value=20)
    _write_rgb(tmp_path / "ir" / "sample.png", value=30)
    _write_rgb(tmp_path / "hazy" / "mist" / "sample.png", value=40)
    Image.fromarray(np.full((7, 9), 50000, dtype=np.uint16), mode="I;16").save(
        tmp_path / "Transmission_Map_GT" / "mist" / "sample.png"
    )
    filenames = ("sample.png", "sample.tiff") if directory == "Transmission_Map_GT/mist" else ("sample.png", "sample.jpg")
    for filename in filenames:
        path = tmp_path / directory / filename
        if directory == "Transmission_Map_GT/mist":
            Image.fromarray(np.full((7, 9), 50000, dtype=np.uint16), mode="I;16").save(path)
        else:
            _write_rgb(path, value=50)

    with pytest.raises(ValueError, match="duplicate stem.*sample"):
        SynthMultiModalDataset(str(tmp_path), train=False, haze_levels=("mist",))


def test_same_stem_in_different_haze_levels_remains_valid(tmp_path):
    for directory in ("clear", "ir", "hazy/mist", "hazy/dense", "Transmission_Map_GT/mist", "Transmission_Map_GT/dense"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    _write_rgb(tmp_path / "clear" / "sample.png", value=20)
    _write_rgb(tmp_path / "ir" / "sample.png", value=30)
    for level in ("mist", "dense"):
        _write_rgb(tmp_path / "hazy" / level / "sample.png", value=40)
        Image.fromarray(np.full((7, 9), 50000, dtype=np.uint16), mode="I;16").save(
            tmp_path / "Transmission_Map_GT" / level / "sample.png"
        )

    assert len(SynthMultiModalDataset(str(tmp_path), train=False, haze_levels=("mist", "dense"))) == 2


def test_synth_training_geometry_preserves_aspect_ratio_and_eval_keeps_original_size(tmp_path):
    for directory in ("clear", "ir", "hazy/mist", "Transmission_Map_GT/mist"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    array = np.arange(63, dtype=np.uint8).reshape(7, 9)
    Image.fromarray(np.stack((array, array, array), axis=-1), mode="RGB").save(tmp_path / "clear" / "sample.png")
    Image.fromarray(np.stack((array, array, array), axis=-1), mode="RGB").save(tmp_path / "ir" / "sample.png")
    Image.fromarray(np.stack((array, array, array), axis=-1), mode="RGB").save(tmp_path / "hazy" / "mist" / "sample.png")
    Image.fromarray(array.astype(np.uint16), mode="I;16").save(tmp_path / "Transmission_Map_GT" / "mist" / "sample.png")

    train = SynthMultiModalDataset(str(tmp_path), train=True, size=8, haze_levels=("mist",), augmentation_seed_base=4)
    desc = train.geometry_description(0, (7, 9))
    train_tensors = train[0]
    evaluation = SynthMultiModalDataset(str(tmp_path), train=False, size=8, haze_levels=("mist",))
    eval_tensors = evaluation[0]

    assert desc["resized_height"] == 8 and desc["resized_width"] == 10
    assert desc["crop_height"] == desc["crop_width"] == 8
    assert all(tensor.shape[-2:] == (8, 8) for tensor in train_tensors)
    assert all(tensor.shape[-2:] == (7, 9) for tensor in eval_tensors)
