import json

import numpy as np
import pytest
import torch
from torch import nn
from PIL import Image


def _write_rgb(path, value):
    Image.new("RGB", (32, 32), (value, value, value)).save(path)


def test_source_training_entrypoint_runs_tiny_batch_and_writes_epoch_checkpoint(tmp_path):
    import Teacher
    Teacher.build_regional_reconstruction_criteria = lambda device: (
        type("OneSSIM", (nn.Module,), {"forward": lambda self, prediction, _clear: prediction.new_ones(())})().to(device),
        type("ZeroContrast", (nn.Module,), {"forward": lambda self, prediction, _clear, _hazy: prediction.mean() * 0})().to(device),
    )
    for directory in ("clear", "ir", "hazy/1_mist", "Transmission_Map_GT/1_mist", "mask_GT/1_mist"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    _write_rgb(tmp_path / "clear" / "sample.png", 90)
    _write_rgb(tmp_path / "ir" / "sample.png", 40)
    _write_rgb(tmp_path / "hazy" / "1_mist" / "sample.png", 120)
    Image.fromarray(np.full((32, 32), 50000, dtype=np.uint16), mode="I;16").save(
        tmp_path / "Transmission_Map_GT" / "1_mist" / "sample.png"
    )
    Image.fromarray(np.ones((32, 32), dtype=np.uint8) * 255, mode="L").save(tmp_path / "mask_GT" / "1_mist" / "sample.png")

    from Teacher import main
    checkpoint_dir = tmp_path / "checkpoints"
    main([
        "--train_data_dir", str(tmp_path), "--validation_data_dir", str(tmp_path), "--train_size", "32", "--epochs", "1", "--iters_per_epoch", "1", "--device", "cpu",
        "--base_channels", "8", "--memory_max_tokens", "16", "--memory_topk", "2",
        "--saved_model_dir", str(checkpoint_dir), "--exp_dir", str(tmp_path / "experiment"),
    ])

    checkpoint = torch.load(checkpoint_dir / "source_last.pt", map_location="cpu")
    assert checkpoint["training_stage"] == "source"
    assert checkpoint["global_step"] == 1
    assert set(checkpoint) == {"format_version", "training_stage", "model", "optimizer", "epoch", "global_step", "config", "best_psnr"}
    assert (checkpoint_dir / "source_best.pt").is_file()
    experiment_dir = tmp_path / "experiment"
    summary = json.loads((experiment_dir / "run_summary.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (experiment_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert summary["stage"] == "source"
    assert summary["res2net_pretrained_loaded"] is True
    assert summary["dataset_sizes"] == {"train": 1, "validation": 1}
    assert {event["event"] for event in events} == {"train_step", "validation"}
    assert "loss_total" in next(event for event in events if event["event"] == "train_step")

    # A checkpoint saved exactly at the one-sample epoch boundary must advance
    # to a fresh deterministic sampler permutation rather than yield no batch.
    main([
        "--train_data_dir", str(tmp_path), "--validation_data_dir", str(tmp_path), "--train_size", "32", "--epochs", "2", "--iters_per_epoch", "1", "--device", "cpu",
        "--base_channels", "8", "--memory_max_tokens", "16", "--memory_topk", "2",
        "--saved_model_dir", str(checkpoint_dir), "--exp_dir", str(tmp_path / "experiment-resume"),
        "--resume_checkpoint", str(checkpoint_dir / "source_last.pt"),
        "--learning_rate", "0.002",
    ])
    resumed = torch.load(checkpoint_dir / "source_last.pt", map_location="cpu")
    assert resumed["global_step"] == 2
    assert resumed["config"]["start_lr"] == 0.002
    assert (checkpoint_dir / "source_best.pt").is_file()
    assert resumed["optimizer"]["param_groups"][0]["lr"] == pytest.approx(1e-6)
