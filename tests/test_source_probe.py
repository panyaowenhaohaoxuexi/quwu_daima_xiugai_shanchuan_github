import torch
import numpy as np
from PIL import Image
from torch import nn

from Teacher import save_source_probe


class _ProbeModel(nn.Module):
    def forward(self, hazy, _tir, *, route_temperature, route_mode):
        assert route_temperature == 0.2
        assert route_mode == "hard"
        route = hazy[:, :1].new_full((hazy.shape[0], 1, *hazy.shape[-2:]), 0.75)
        return {
            "pred_clear": hazy * 0.5,
            "density_map": route * 0.5,
            "route_soft": route,
            "route_hard": torch.ones_like(route),
            "boundary_map": torch.zeros_like(route),
        }


def test_source_probe_saves_real_rgb_tir_predictions_without_any_ground_truth(tmp_path):
    hazy_path, tir_path = tmp_path / "hazy.png", tmp_path / "tir.png"
    Image.new("RGB", (10, 8), (120, 110, 100)).save(hazy_path)
    Image.new("RGB", (10, 8), (50, 50, 50)).save(tir_path)

    metrics = save_source_probe(
        _ProbeModel(), hazy_path, tir_path, tmp_path / "outputs", device=torch.device("cpu"),
        route_temperature=0.2, pair_alignment_policy="strict", tir_normalization_config={}, prefix="epoch_0001",
    )

    assert metrics == {"density_mean": 0.375, "route_hard_fraction": 1.0}
    for name in ("pred_clear", "density_map", "route_soft", "route_hard", "boundary_map"):
        assert (tmp_path / "outputs" / f"epoch_0001_{name}.png").is_file()


def test_source_training_saves_real_probe_only_when_validation_psnr_improves(tmp_path, monkeypatch):
    import Teacher

    for directory in ("clear", "ir", "hazy/1_mist", "Transmission_Map_GT/1_mist", "mask_GT/1_mist"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), (90, 90, 90)).save(tmp_path / "clear" / "sample.png")
    Image.new("RGB", (32, 32), (40, 40, 40)).save(tmp_path / "ir" / "sample.png")
    Image.new("RGB", (32, 32), (120, 120, 120)).save(tmp_path / "hazy" / "1_mist" / "sample.png")
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
    scores = iter(({"psnr": 20.0, "ssim": 0.8}, {"psnr": 19.0, "ssim": 0.7}))
    saved = []
    monkeypatch.setattr(Teacher, "build_regional_reconstruction_criteria", criteria)
    monkeypatch.setattr(Teacher, "evaluate_paired_validation", lambda *_args, **_kwargs: next(scores))
    monkeypatch.setattr(Teacher, "save_source_probe", lambda *_args, **kwargs: saved.append(kwargs["prefix"]) or {
        "density_mean": 0.1, "route_hard_fraction": 0.2,
    })

    Teacher.main([
        "--train_data_dir", str(tmp_path), "--validation_data_dir", str(tmp_path), "--train_size", "32",
        "--epochs", "2", "--iters_per_epoch", "1", "--device", "cpu", "--num_workers", "0",
        "--base_channels", "8", "--memory_max_tokens", "16", "--memory_topk", "2",
        "--saved_model_dir", str(tmp_path / "checkpoints"), "--exp_dir", str(tmp_path / "experiment"),
        "--source_probe_hazy", "probe_hazy.png", "--source_probe_tir", "probe_tir.png",
        "--source_probe_output_dir", str(tmp_path / "probe"),
    ])

    assert saved == ["source_best_epoch_0001_step_00000001"]
