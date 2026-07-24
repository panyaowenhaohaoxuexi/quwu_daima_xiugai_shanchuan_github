import torch
from PIL import Image

from Eval import main
from model import FogRoutedRGBTIRDehazer


def test_eval_saves_prediction_and_aux_at_original_size(tmp_path):
    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2)
    config = {
        "base_channels": 8, "router_hidden_channels": 8, "deform_num_samples": 4,
        "deform_max_offset": 2.0, "num_structure_renderers": 2, "memory_max_tokens": 16,
        "memory_topk": 2, "memory_attention_temperature": 0.07,
        "memory_reliability_epsilon": 1e-6, "memory_reliable_ratio_threshold": 0.01,
        "memory_confidence_threshold": 0.1, "memory_exclusion_extra_margin": 0,
        "boundary_width": 1, "route_tau_end": 0.2,
        "tir_normalization": "dtype_range", "tir_percentile_low": 1.0,
        "tir_percentile_high": 99.0, "tir_percentile_scope": "per_image",
        "tir_channel_tolerance_code_values": 1, "tir_channel_tolerance_float": 1e-5,
    }
    checkpoint = tmp_path / "source.pt"
    torch.save({"format_version": 1, "training_stage": "source", "model_class": "FogRoutedRGBTIRDehazer",
                "density_gt_semantics": "transmission", "config": config, "model": model.state_dict()}, checkpoint)
    hazy_dir, tir_dir, out_dir = tmp_path / "hazy", tmp_path / "tir", tmp_path / "out"
    hazy_dir.mkdir(); tir_dir.mkdir()
    Image.new("RGB", (47, 31), (80, 80, 80)).save(hazy_dir / "x.png")
    Image.new("L", (47, 31), 100).save(tir_dir / "x.png")

    main(["--checkpoint", str(checkpoint), "--hazy_dir", str(hazy_dir), "--tir_dir", str(tir_dir),
          "--output_dir", str(out_dir), "--device", "cpu", "--save_aux"])

    assert Image.open(out_dir / "x.png").size == (47, 31)
    assert (out_dir / "aux" / "x_density_map.png").is_file()


def test_eval_rejects_mismatched_tir_under_strict_checkpoint_alignment(tmp_path):
    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2)
    config = {
        "base_channels": 8, "router_hidden_channels": 8, "deform_num_samples": 4,
        "deform_max_offset": 2.0, "num_structure_renderers": 2, "memory_max_tokens": 16,
        "memory_topk": 2, "memory_attention_temperature": 0.07,
        "memory_reliability_epsilon": 1e-6, "memory_reliable_ratio_threshold": 0.01,
        "memory_confidence_threshold": 0.1, "memory_exclusion_extra_margin": 0,
        "boundary_width": 1, "route_tau_end": 0.2, "pair_alignment_policy": "strict",
        "tir_normalization": "dtype_range", "tir_percentile_low": 1.0,
        "tir_percentile_high": 99.0, "tir_percentile_scope": "per_image",
        "tir_channel_tolerance_code_values": 1, "tir_channel_tolerance_float": 1e-5,
    }
    checkpoint = tmp_path / "source.pt"
    torch.save({"format_version": 1, "training_stage": "source", "model_class": "FogRoutedRGBTIRDehazer",
                "density_gt_semantics": "transmission", "config": config, "model": model.state_dict()}, checkpoint)
    hazy_dir, tir_dir, out_dir = tmp_path / "hazy", tmp_path / "tir", tmp_path / "out"
    hazy_dir.mkdir(); tir_dir.mkdir()
    Image.new("RGB", (47, 31), (80, 80, 80)).save(hazy_dir / "x.png")
    Image.new("L", (20, 20), 100).save(tir_dir / "x.png")

    import pytest
    with pytest.raises(ValueError, match="pair alignment failed"):
        main(["--checkpoint", str(checkpoint), "--hazy_dir", str(hazy_dir), "--tir_dir", str(tir_dir),
              "--output_dir", str(out_dir), "--device", "cpu"])


def test_eval_honors_requested_output_format(tmp_path):
    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2)
    config = {
        "base_channels": 8, "router_hidden_channels": 8, "deform_num_samples": 4,
        "deform_max_offset": 2.0, "num_structure_renderers": 2, "memory_max_tokens": 16,
        "memory_topk": 2, "memory_attention_temperature": 0.07,
        "memory_reliability_epsilon": 1e-6, "memory_reliable_ratio_threshold": 0.01,
        "memory_confidence_threshold": 0.1, "memory_exclusion_extra_margin": 0,
        "boundary_width": 1, "route_tau_end": 0.2, "pair_alignment_policy": "strict",
        "tir_normalization": "dtype_range", "tir_percentile_low": 1.0,
        "tir_percentile_high": 99.0, "tir_percentile_scope": "per_image",
        "tir_channel_tolerance_code_values": 1, "tir_channel_tolerance_float": 1e-5,
    }
    checkpoint = tmp_path / "source.pt"
    torch.save({"format_version": 1, "training_stage": "source", "model_class": "FogRoutedRGBTIRDehazer",
                "density_gt_semantics": "transmission", "config": config, "model": model.state_dict()}, checkpoint)
    hazy_dir, tir_dir, out_dir = tmp_path / "hazy", tmp_path / "tir", tmp_path / "out"
    hazy_dir.mkdir(); tir_dir.mkdir()
    Image.new("RGB", (32, 32), (80, 80, 80)).save(hazy_dir / "x.png")
    Image.new("L", (32, 32), 100).save(tir_dir / "x.png")

    main(["--checkpoint", str(checkpoint), "--hazy_dir", str(hazy_dir), "--tir_dir", str(tir_dir),
          "--output_dir", str(out_dir), "--device", "cpu", "--format", "bmp"])
    assert (out_dir / "x.bmp").is_file()
