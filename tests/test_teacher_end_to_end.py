import numpy as np
import torch
from PIL import Image


def _write_rgb(path, value):
    Image.new("RGB", (32, 32), (value, value, value)).save(path)


def test_source_training_entrypoint_runs_tiny_batch_and_writes_strict_checkpoint(tmp_path):
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
    assert checkpoint["format_version"] == 1
    assert checkpoint["training_stage"] == "source"
    assert checkpoint["model_class"] == "FogRoutedRGBTIRDehazer"
    assert checkpoint["global_step"] == 1
    assert checkpoint["density_gt_semantics"] == "transmission"
    assert checkpoint["rng_state"]["omega_generator"] is not None

    # A checkpoint saved exactly at the one-sample epoch boundary must advance
    # to a fresh deterministic sampler permutation rather than yield no batch.
    main([
        "--train_data_dir", str(tmp_path), "--train_size", "32", "--epochs", "2", "--device", "cpu",
        "--base_channels", "8", "--memory_max_tokens", "16", "--memory_topk", "2",
        "--counterfactual_start_step", "100", "--route_loss_start_step", "100",
        "--saved_model_dir", str(checkpoint_dir), "--exp_dir", str(tmp_path / "experiment-resume"),
        "--resume_checkpoint", str(checkpoint_dir / "source_last.pt"),
    ])
    resumed = torch.load(checkpoint_dir / "source_last.pt", map_location="cpu")
    assert resumed["global_step"] == 2


def test_source_failure_retries_with_identical_omega_and_loader_generator_state(tmp_path, monkeypatch):
    for directory in ("clear", "ir", "hazy/mist", "Transmission_Map_GT/mist"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    _write_rgb(tmp_path / "clear" / "sample.png", 90)
    _write_rgb(tmp_path / "ir" / "sample.png", 40)
    _write_rgb(tmp_path / "hazy" / "mist" / "sample.png", 120)
    Image.fromarray(np.full((32, 32), 50000, dtype=np.uint16), mode="I;16").save(
        tmp_path / "Transmission_Map_GT" / "mist" / "sample.png"
    )
    import Teacher

    common = [
        "--train_data_dir", str(tmp_path), "--train_size", "32", "--epochs", "1", "--device", "cpu",
        "--base_channels", "8", "--memory_max_tokens", "16", "--memory_topk", "2",
        "--counterfactual_start_step", "100", "--route_loss_start_step", "100",
    ]
    reference_dir = tmp_path / "reference"
    Teacher.main([*common, "--saved_model_dir", str(reference_dir), "--exp_dir", str(tmp_path / "reference-exp")])
    reference = torch.load(reference_dir / "source_last.pt", map_location="cpu")
    original = Teacher.perform_optimizer_step
    calls = {"count": 0}

    def fail_once(*args, **kwargs):
        calls["count"] += 1
        return False if calls["count"] == 1 else original(*args, **kwargs)

    monkeypatch.setattr(Teacher, "perform_optimizer_step", fail_once)
    retry_dir = tmp_path / "retry"
    Teacher.main([*common, "--saved_model_dir", str(retry_dir), "--exp_dir", str(tmp_path / "retry-exp")])
    retried = torch.load(retry_dir / "source_last.pt", map_location="cpu")

    assert calls["count"] == 2
    assert retried["global_step"] == 1 and retried["epoch"] == 1
    assert torch.equal(retried["rng_state"]["omega_generator"], reference["rng_state"]["omega_generator"])
    assert torch.equal(
        retried["rng_state"]["dataloader_generators"]["source"],
        reference["rng_state"]["dataloader_generators"]["source"],
    )
