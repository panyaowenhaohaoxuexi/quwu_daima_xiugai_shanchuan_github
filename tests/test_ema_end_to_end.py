import numpy as np
import pytest
import torch
from torch import nn
from PIL import Image


def _write_rgb(path, value):
    Image.new("RGB", (32, 32), (value, value, value)).save(path)


@pytest.fixture(autouse=True)
def _mock_coa_clip_for_cpu_ema_smoke(monkeypatch):
    import EMA

    class _ZeroClipLoss(nn.Module):
        def forward(self, prediction, _text_features):
            return prediction.mean() * 0

    monkeypatch.setattr(
        EMA,
        "initialize_coa_clip",
        lambda device: (_ZeroClipLoss().to(device), torch.zeros(1, 1, device=device)),
    )
    monkeypatch.setattr(EMA, "require_coa_clip_cuda", lambda _device: None)


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
        "format_version", "training_stage", "student", "teacher", "optimizer", "epoch",
        "source_global_step", "ema_global_step", "config",
    }

    main([
        "--resume_checkpoint", str(checkpoint_dir / "ema_last.pt"), "--source_anchor_data_dir", str(tmp_path),
        "--real_data_dir", str(tmp_path / "real"), "--epochs", "2", "--device", "cpu",
        "--learning_rate", "0.003", "--saved_model_dir", str(checkpoint_dir),
        "--exp_dir", str(tmp_path / "ema-experiment-resume"),
    ])
    resumed = torch.load(checkpoint_dir / "ema_last.pt", map_location="cpu")
    assert resumed["ema_global_step"] == 2
    assert resumed["config"]["learning_rate"] == 0.003
    assert resumed["optimizer"]["param_groups"][0]["lr"] == 0.003


def test_ema_lambda_router_zero_does_not_raise_for_empty_q_regions(tmp_path):
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
    source_dir = tmp_path / "source"
    source_main([
        "--train_data_dir", str(tmp_path), "--train_size", "32", "--epochs", "1", "--device", "cpu",
        "--base_channels", "8", "--memory_max_tokens", "16", "--memory_topk", "2",
        "--counterfactual_start_step", "100", "--route_loss_start_step", "100",
        "--lambda_router", "0", "--q_min_valid_support", "64",
        "--max_consecutive_empty_omega_steps", "1", "--saved_model_dir", str(source_dir),
        "--exp_dir", str(tmp_path / "source-exp"),
    ])

    from EMA import main
    ema_dir = tmp_path / "ema"
    main([
        "--source_checkpoint", str(source_dir / "source_last.pt"), "--source_anchor_data_dir", str(tmp_path),
        "--real_data_dir", str(tmp_path / "real"), "--epochs", "1", "--device", "cpu",
        "--saved_model_dir", str(ema_dir), "--exp_dir", str(tmp_path / "ema-exp"),
    ])
    assert (ema_dir / "ema_last.pt").is_file()
