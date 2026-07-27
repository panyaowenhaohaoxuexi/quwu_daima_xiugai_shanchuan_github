import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image


def _write_rgb(path, value):
    Image.new("RGB", (32, 32), (value, value, value)).save(path)


def _write_minimal_training_data(root):
    for directory in ("clear", "ir", "hazy/mist", "Transmission_Map_GT/mist", "real/hazy", "real/tir"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    _write_rgb(root / "clear" / "sample.png", 90)
    _write_rgb(root / "ir" / "sample.png", 40)
    _write_rgb(root / "hazy" / "mist" / "sample.png", 120)
    Image.fromarray(np.full((32, 32), 50000, dtype=np.uint16), mode="I;16").save(
        root / "Transmission_Map_GT" / "mist" / "sample.png"
    )
    _write_rgb(root / "real" / "hazy" / "real.png", 100)
    _write_rgb(root / "real" / "tir" / "real.png", 55)


def test_ema_seed_function_repeats_python_numpy_and_torch_sequences():
    from EMA import _set_seed

    _set_seed(37)
    first = (random.random(), float(np.random.rand()), float(torch.rand(())))
    _set_seed(37)
    assert (random.random(), float(np.random.rand()), float(torch.rand(()))) == first


def test_ema_startup_uses_cpu_geometry_generator_before_model_construction():
    source = Path("EMA.py").read_text(encoding="utf-8")

    assert "geometry_generator = torch.Generator().manual_seed(" in source
    assert "omega_generator = torch.Generator(device=device).manual_seed(" in source
    assert source.index("_set_seed(args.model_init_seed)") < source.index("student = build_model_from_config")
    assert source.index("student = build_model_from_config") < source.index("optimizer = AdamW")
    assert source.index("optimizer = AdamW") < source.index("geometry_generator = torch.Generator()")
    assert source.index("omega_generator = torch.Generator(device=device)") < source.index("real_dataset = RealMultiModalDataset")


def test_cpu_geometry_generator_produces_three_consecutive_transforms():
    from EMA import _sample_geometry

    generator = torch.Generator().manual_seed(202)
    transforms = tuple(_sample_geometry(generator) for _ in range(3))

    assert len(transforms) == 3
    assert all(0 <= item.rot90_k < 4 and isinstance(item.horizontal_flip, bool) for item in transforms)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_ema_cuda_smoke_has_no_geometry_generator_device_mismatch(tmp_path):
    _write_minimal_training_data(tmp_path)
    from Teacher import main as source_main
    from EMA import main as ema_main

    source_dir, ema_dir = tmp_path / "source", tmp_path / "ema"
    source_main([
        "--train_data_dir", str(tmp_path), "--train_size", "32", "--epochs", "1", "--device", "cpu",
        "--base_channels", "8", "--memory_max_tokens", "16", "--memory_topk", "2",
        "--counterfactual_start_step", "100", "--route_loss_start_step", "100",
        "--saved_model_dir", str(source_dir), "--exp_dir", str(tmp_path / "source-exp"),
    ])
    ema_main([
        "--source_checkpoint", str(source_dir / "source_last.pt"), "--source_anchor_data_dir", str(tmp_path),
        "--real_data_dir", str(tmp_path / "real"), "--epochs", "1", "--device", "cuda",
        "--saved_model_dir", str(ema_dir), "--exp_dir", str(tmp_path / "ema-exp"),
    ])
    assert (ema_dir / "ema_last.pt").is_file()
