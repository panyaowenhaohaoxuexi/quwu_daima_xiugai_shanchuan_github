import numpy as np
import torch
from PIL import Image


def _write_rgb(path, value):
    Image.new("RGB", (32, 32), (value, value, value)).save(path)


def test_ema_entrypoint_runs_real_and_source_anchor_from_source_checkpoint(tmp_path):
    for directory in ("clear", "ir", "hazy/mist", "Transmission_Map_GT/mist", "real/hazy", "real/tir"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    _write_rgb(tmp_path / "clear" / "sample.png", 90)
    _write_rgb(tmp_path / "ir" / "sample.png", 40)
    _write_rgb(tmp_path / "hazy" / "mist" / "sample.png", 120)
    Image.fromarray(np.full((32, 32), 50000, dtype=np.uint16), mode="I;16").save(
        tmp_path / "Transmission_Map_GT" / "mist" / "sample.png"
    )
    _write_rgb(tmp_path / "real" / "hazy" / "real.png", 100)
    _write_rgb(tmp_path / "real" / "tir" / "real.png", 55)

    from Teacher import main as source_main
    source_dir = tmp_path / "source-checkpoint"
    source_main([
        "--train_data_dir", str(tmp_path), "--train_size", "32", "--epochs", "1", "--device", "cpu",
        "--base_channels", "8", "--memory_max_tokens", "16", "--memory_topk", "2",
        "--counterfactual_start_step", "100", "--route_loss_start_step", "100",
        "--saved_model_dir", str(source_dir), "--exp_dir", str(tmp_path / "source-exp"),
    ])

    from EMA import main
    checkpoint_dir = tmp_path / "ema-checkpoints"
    main([
        "--source_checkpoint", str(source_dir / "source_last.pt"), "--source_anchor_data_dir", str(tmp_path),
        "--real_data_dir", str(tmp_path / "real"), "--epochs", "1", "--device", "cpu",
        "--saved_model_dir", str(checkpoint_dir), "--exp_dir", str(tmp_path / "ema-experiment"),
    ])
    checkpoint = torch.load(checkpoint_dir / "ema_last.pt", map_location="cpu")
    assert checkpoint["training_stage"] == "ema"
    assert checkpoint["ema_global_step"] == 1
    assert set(checkpoint) == {
        "training_stage", "student", "teacher", "optimizer", "epoch",
        "source_global_step", "ema_global_step", "config",
    }
