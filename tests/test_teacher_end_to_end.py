import numpy as np
import torch
from PIL import Image


def _write_rgb(path, value):
    Image.new("RGB", (32, 32), (value, value, value)).save(path)


def test_source_training_entrypoint_runs_tiny_batch_and_writes_epoch_checkpoint(tmp_path):
    for directory in ("clear", "ir", "hazy/mist", "Transmission_Map_GT/mist"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    _write_rgb(tmp_path / "clear" / "sample.png", 90)
    _write_rgb(tmp_path / "ir" / "sample.png", 40)
    _write_rgb(tmp_path / "hazy" / "mist" / "sample.png", 120)
    Image.fromarray(np.full((32, 32), 50000, dtype=np.uint16), mode="I;16").save(
        tmp_path / "Transmission_Map_GT" / "mist" / "sample.png"
    )

    from Teacher import main
    checkpoint_dir = tmp_path / "checkpoints"
    main([
        "--train_data_dir", str(tmp_path), "--train_size", "32", "--epochs", "1", "--device", "cpu",
        "--base_channels", "8", "--memory_max_tokens", "16", "--memory_topk", "2",
        "--counterfactual_start_step", "100", "--route_loss_start_step", "100",
        "--saved_model_dir", str(checkpoint_dir), "--exp_dir", str(tmp_path / "experiment"),
    ])

    checkpoint = torch.load(checkpoint_dir / "source_last.pt", map_location="cpu")
    assert checkpoint["training_stage"] == "source"
    assert checkpoint["global_step"] == 1
    assert set(checkpoint) == {"training_stage", "model", "optimizer", "epoch", "global_step", "config"}

    # A checkpoint saved exactly at the one-sample epoch boundary must advance
    # to a fresh deterministic sampler permutation rather than yield no batch.
    main([
        "--train_data_dir", str(tmp_path), "--train_size", "32", "--epochs", "2", "--device", "cpu",
        "--base_channels", "8", "--memory_max_tokens", "16", "--memory_topk", "2",
        "--counterfactual_start_step", "100", "--route_loss_start_step", "100",
        "--saved_model_dir", str(checkpoint_dir), "--exp_dir", str(tmp_path / "experiment-resume"),
        "--resume_checkpoint", str(checkpoint_dir / "source_last.pt"),
        "--learning_rate", "0.002",
    ])
    resumed = torch.load(checkpoint_dir / "source_last.pt", map_location="cpu")
    assert resumed["global_step"] == 2
    assert resumed["config"]["learning_rate"] == 0.002
    assert resumed["optimizer"]["param_groups"][0]["lr"] == 0.002
