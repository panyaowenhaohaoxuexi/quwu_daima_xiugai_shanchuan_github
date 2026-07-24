import numpy as np
import torch
from PIL import Image


def _write_rgb(path, value):
    Image.new("RGB", (32, 32), (value, value, value)).save(path)


def test_ema_entrypoint_runs_real_and_source_anchor_from_strict_source_checkpoint(tmp_path):
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

    from model import FogRoutedRGBTIRDehazer
    from option.Teacher import build_parser as build_source_parser
    from training.checkpointing import build_source_checkpoint, capture_rng_state
    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2)
    source_args = build_source_parser().parse_args([
        "--base_channels", "8", "--memory_max_tokens", "16", "--memory_topk", "2", "--train_size", "32",
    ])
    source_checkpoint = tmp_path / "source.pt"
    torch.save(build_source_checkpoint(
        model.state_dict(), torch.optim.AdamW(model.parameters()).state_dict(), None, 0, 0,
        vars(source_args), "transmission", capture_rng_state(),
    ), source_checkpoint)

    from EMA import main
    checkpoint_dir = tmp_path / "ema-checkpoints"
    main([
        "--source_checkpoint", str(source_checkpoint), "--train_data_dir", str(tmp_path),
        "--real_data_dir", str(tmp_path / "real"), "--train_size", "32", "--epochs", "1", "--device", "cpu",
        "--saved_model_dir", str(checkpoint_dir), "--exp_dir", str(tmp_path / "ema-experiment"),
        "--counterfactual_chunk_size", "6",
    ])
    checkpoint = torch.load(checkpoint_dir / "ema_last.pt", map_location="cpu")
    assert checkpoint["training_stage"] == "ema"
    assert checkpoint["ema_global_step"] == 1
    assert {"student", "teacher", "source_global_step", "ema_global_step"}.issubset(checkpoint)
    assert checkpoint["rng_state"]["omega_generator"] is not None
    assert checkpoint["rng_state"]["geometry_generator"] is not None

    # Both independent real/source cursors are at epoch boundaries after one
    # successful step. EMA resume must advance both and perform the next step.
    main([
        "--resume_checkpoint", str(checkpoint_dir / "ema_last.pt"), "--train_data_dir", str(tmp_path),
        "--real_data_dir", str(tmp_path / "real"), "--train_size", "32", "--epochs", "2", "--device", "cpu",
        "--saved_model_dir", str(checkpoint_dir), "--exp_dir", str(tmp_path / "ema-resume"),
        "--counterfactual_chunk_size", "6",
    ])
    resumed = torch.load(checkpoint_dir / "ema_last.pt", map_location="cpu")
    assert resumed["ema_global_step"] == 2


def test_ema_failure_retries_with_identical_loader_generators(tmp_path, monkeypatch):
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
    from model import FogRoutedRGBTIRDehazer
    from option.Teacher import build_parser as build_source_parser
    from training.checkpointing import build_source_checkpoint, capture_rng_state
    import EMA

    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2)
    source_args = build_source_parser().parse_args([
        "--base_channels", "8", "--memory_max_tokens", "16", "--memory_topk", "2", "--train_size", "32",
    ])
    source_checkpoint = tmp_path / "source.pt"
    torch.save(build_source_checkpoint(
        model.state_dict(), torch.optim.AdamW(model.parameters()).state_dict(), None, 0, 0,
        vars(source_args), "transmission", capture_rng_state(),
    ), source_checkpoint)
    common = [
        "--source_checkpoint", str(source_checkpoint), "--train_data_dir", str(tmp_path),
        "--real_data_dir", str(tmp_path / "real"), "--train_size", "32", "--epochs", "1", "--device", "cpu",
        "--counterfactual_chunk_size", "6",
    ]
    reference_dir = tmp_path / "reference"
    EMA.main([*common, "--saved_model_dir", str(reference_dir), "--exp_dir", str(tmp_path / "reference-exp")])
    reference = torch.load(reference_dir / "ema_last.pt", map_location="cpu")
    original = EMA.run_ema_adaptation_step
    calls = {"count": 0}

    def fail_once(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"L_real": torch.tensor(0.0), "L_src": torch.tensor(0.0), "L_adapt": torch.tensor(0.0), "step_succeeded": False}
        return original(*args, **kwargs)

    monkeypatch.setattr(EMA, "run_ema_adaptation_step", fail_once)
    retry_dir = tmp_path / "retry"
    EMA.main([*common, "--saved_model_dir", str(retry_dir), "--exp_dir", str(tmp_path / "retry-exp")])
    retried = torch.load(retry_dir / "ema_last.pt", map_location="cpu")

    assert calls["count"] == 2
    assert retried["ema_global_step"] == 1 and retried["epoch"] == 1
    for name in ("real", "source_anchor"):
        assert torch.equal(
            retried["rng_state"]["dataloader_generators"][name],
            reference["rng_state"]["dataloader_generators"][name],
        )
