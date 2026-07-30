import torch

from UDA import build_target_reference_loader, route_consistency_multiplier, style_source_batch


def test_stage_a_target_reference_loader_discards_only_incomplete_multi_image_batch():
    dataset = [(torch.zeros(3, 8, 8), torch.zeros(3, 8, 8), {"sample_id": str(index)}) for index in range(5)]

    multi_image_batches = list(build_target_reference_loader(dataset, batch_size=2, num_workers=0, drop_last=True))
    single_image_batches = list(build_target_reference_loader(dataset, batch_size=1, num_workers=0, drop_last=False))

    assert [hazy.shape[0] for hazy, _tir, _metadata in multi_image_batches] == [2, 2]
    assert [hazy.shape[0] for hazy, _tir, _metadata in single_image_batches] == [1, 1, 1, 1, 1]


def test_style_source_batch_preserves_physical_labels_and_can_style_every_sample():
    source = (
        torch.rand(2, 3, 8, 8),
        torch.rand(2, 3, 8, 8),
        torch.rand(2, 3, 8, 8),
        torch.rand(2, 1, 8, 8),
        torch.randint(0, 2, (2, 1, 8, 8)).float(),
    )
    target_hazy, target_tir = torch.rand(2, 3, 10, 12), torch.rand(2, 3, 10, 12)

    styled, applied = style_source_batch(
        source, target_hazy, target_tir, generator=torch.Generator().manual_seed(3),
        probability=1.0, beta_min=1.0, beta_max=1.0,
        min_gain=0.75, max_gain=1.35, max_abs_bias=0.20,
    )

    assert applied.all()
    assert torch.equal(styled[3], source[3])
    assert torch.equal(styled[4], source[4])
    assert not torch.equal(styled[0], source[0])


def test_route_consistency_multiplier_has_warmup_then_linear_ramp():
    assert route_consistency_multiplier(0, warmup_steps=10, ramp_steps=20) == 0.0
    assert route_consistency_multiplier(10, warmup_steps=10, ramp_steps=20) == 0.0
    assert route_consistency_multiplier(20, warmup_steps=10, ramp_steps=20) == 0.5
    assert route_consistency_multiplier(30, warmup_steps=10, ramp_steps=20) == 1.0


def test_stage_a_entrypoint_writes_a_source_checkpoint(tmp_path, monkeypatch):
    """Stage A keeps Source supervision but writes a self-contained new anchor."""
    import numpy as np
    from PIL import Image
    from torch import nn
    import Teacher
    import UDA

    def write_rgb(path, value):
        Image.new("RGB", (32, 32), (value, value, value)).save(path)

    for directory in ("clear", "ir", "hazy/1_mist", "Transmission_Map_GT/1_mist",
                      "mask_GT/1_mist", "real/hazy", "real/tir"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    write_rgb(tmp_path / "clear" / "sample.png", 90)
    write_rgb(tmp_path / "ir" / "sample.png", 40)
    write_rgb(tmp_path / "hazy" / "1_mist" / "sample.png", 120)
    write_rgb(tmp_path / "real" / "hazy" / "real.png", 100)
    write_rgb(tmp_path / "real" / "tir" / "real.png", 55)
    Image.fromarray(np.full((32, 32), 50000, dtype=np.uint16), mode="I;16").save(
        tmp_path / "Transmission_Map_GT" / "1_mist" / "sample.png"
    )
    Image.fromarray(np.full((32, 32), 255, dtype=np.uint8), mode="L").save(
        tmp_path / "mask_GT" / "1_mist" / "sample.png"
    )
    criteria = lambda device: (
        type("OneSSIM", (nn.Module,), {"forward": lambda self, prediction, _clear: prediction.new_ones(())})().to(device),
        type("ZeroContrast", (nn.Module,), {"forward": lambda self, prediction, _clear, _hazy: prediction.mean() * 0})().to(device),
    )
    monkeypatch.setattr(Teacher, "build_regional_reconstruction_criteria", criteria)
    monkeypatch.setattr(UDA, "build_regional_reconstruction_criteria", criteria)
    source_dir = tmp_path / "source"
    Teacher.main(["--train_data_dir", str(tmp_path), "--validation_data_dir", str(tmp_path),
                  "--train_size", "32", "--epochs", "1", "--iters_per_epoch", "1", "--device", "cpu",
                  "--base_channels", "8", "--memory_max_tokens", "16", "--memory_topk", "2",
                  "--saved_model_dir", str(source_dir), "--exp_dir", str(tmp_path / "source-exp")])
    uda_dir = tmp_path / "uda"
    probe_dir = tmp_path / "probe"
    UDA.main(["--stage", "source_style", "--source_checkpoint", str(source_dir / "source_last.pt"),
                  "--source_anchor_data_dir", str(tmp_path), "--validation_data_dir", str(tmp_path),
                  "--real_data_dir", str(tmp_path / "real"), "--real_tir_dir", str(tmp_path / "real" / "tir"),
                  "--epochs", "1", "--iters_per_epoch", "1", "--device", "cpu", "--style_probability", "1.0",
                  "--style_beta_min", "1.0", "--style_beta_max", "1.0", "--saved_model_dir", str(uda_dir),
                  "--source_anchor_batch_size", "1", "--real_batch_size", "1", "--num_workers", "0",
                  "--exp_dir", str(tmp_path / "uda-exp"), "--probe_hazy", str(tmp_path / "real" / "hazy" / "real.png"),
              "--probe_tir", str(tmp_path / "real" / "tir" / "real.png"), "--probe_output_dir", str(probe_dir),
              "--saved_data_dir", str(tmp_path / "uda-diagnostics")])
    checkpoint = torch.load(uda_dir / "source_style_last.pt", map_location="cpu")
    assert checkpoint["training_stage"] == "source"
    # Stage A continues the Source route/temperature schedule rather than resetting it.
    assert checkpoint["global_step"] == 2
    for name in ("pred_clear", "density_map", "route_soft", "route_hard", "boundary_map"):
        assert (probe_dir / f"source_style_initial_{name}.png").is_file()


def test_stage_b_entrypoint_delays_real_route_consistency(tmp_path, monkeypatch):
    """The first EMA step is valid with a zero real-route consistency weight."""
    import json
    import numpy as np
    from PIL import Image
    from torch import nn
    import Teacher
    import UDA
    import EMA

    def write_rgb(path, value):
        Image.new("RGB", (32, 32), (value, value, value)).save(path)

    for directory in ("clear", "ir", "hazy/1_mist", "Transmission_Map_GT/1_mist",
                      "mask_GT/1_mist", "real/hazy", "real/tir"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    write_rgb(tmp_path / "clear" / "sample.png", 90); write_rgb(tmp_path / "ir" / "sample.png", 40)
    write_rgb(tmp_path / "hazy" / "1_mist" / "sample.png", 120)
    write_rgb(tmp_path / "real" / "hazy" / "real.png", 100); write_rgb(tmp_path / "real" / "tir" / "real.png", 55)
    Image.fromarray(np.full((32, 32), 50000, dtype=np.uint16), mode="I;16").save(
        tmp_path / "Transmission_Map_GT" / "1_mist" / "sample.png"
    )
    Image.fromarray(np.full((32, 32), 255, dtype=np.uint8), mode="L").save(
        tmp_path / "mask_GT" / "1_mist" / "sample.png"
    )
    criteria = lambda device: (
        type("OneSSIM", (nn.Module,), {"forward": lambda self, prediction, _clear: prediction.new_ones(())})().to(device),
        type("ZeroContrast", (nn.Module,), {"forward": lambda self, prediction, _clear, _hazy: prediction.mean() * 0})().to(device),
    )
    class ZeroClip(nn.Module):
        def forward(self, prediction, _text): return prediction.mean() * 0
    monkeypatch.setattr(Teacher, "build_regional_reconstruction_criteria", criteria)
    monkeypatch.setattr(UDA, "build_regional_reconstruction_criteria", criteria)
    monkeypatch.setattr(EMA, "require_coa_clip_cuda", lambda _device: None)
    monkeypatch.setattr(EMA, "initialize_coa_clip", lambda device: (ZeroClip().to(device), torch.zeros(1, 1, device=device)))
    source_dir = tmp_path / "source"
    Teacher.main(["--train_data_dir", str(tmp_path), "--validation_data_dir", str(tmp_path), "--train_size", "32",
                  "--epochs", "1", "--iters_per_epoch", "1", "--device", "cpu", "--base_channels", "8",
                  "--memory_max_tokens", "16", "--memory_topk", "2", "--saved_model_dir", str(source_dir),
                  "--exp_dir", str(tmp_path / "source-exp")])
    uda_dir = tmp_path / "uda-ema"
    exp_dir = tmp_path / "uda-ema-exp"
    UDA.main(["--stage", "ema", "--source_checkpoint", str(source_dir / "source_last.pt"),
              "--source_anchor_data_dir", str(tmp_path), "--validation_data_dir", str(tmp_path),
              "--real_data_dir", str(tmp_path / "real"), "--real_tir_dir", str(tmp_path / "real" / "tir"),
              "--epochs", "1", "--iters_per_epoch", "1", "--device", "cpu", "--saved_model_dir", str(uda_dir),
              "--source_anchor_batch_size", "1", "--real_batch_size", "1", "--num_workers", "0",
              "--exp_dir", str(exp_dir), "--saved_data_dir", str(tmp_path / "uda-ema-diagnostics"),
              "--route_consistency_warmup_steps", "10", "--probe_hazy", "", "--probe_tir", ""])
    checkpoint = torch.load(uda_dir / "uda_ema_last.pt", map_location="cpu")
    assert checkpoint["training_stage"] == "ema"
    steps = [json.loads(line) for line in (exp_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()]
    assert next(record for record in steps if record["event"] == "train_step")["route_consistency_multiplier"] == 0.0
