import torch
import pytest
from PIL import Image

from Eval import main
from model import FogRoutedRGBTIRDehazer


def test_eval_saves_prediction_and_aux_at_original_size(tmp_path):
    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2)
    config = {
        "base_channels": 8, "router_hidden_channels": 8, "deform_num_samples": 4,
        "deform_max_offset": 2.0, "num_structure_renderers": 2, "memory_max_tokens": 16,
        "memory_topk": 2, "memory_query_chunk_size": 1024, "memory_attention_temperature": 0.07,
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
        "memory_topk": 2, "memory_query_chunk_size": 1024, "memory_attention_temperature": 0.07,
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
        "memory_topk": 2, "memory_query_chunk_size": 1024, "memory_attention_temperature": 0.07,
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


def test_eval_auto_detects_ema_and_selects_student_or_teacher_weights(tmp_path):
    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2)
    teacher = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2)
    config = {
        "base_channels": 8, "router_hidden_channels": 8, "deform_num_samples": 4,
        "deform_max_offset": 2.0, "num_structure_renderers": 2, "memory_max_tokens": 16,
        "memory_topk": 2, "memory_query_chunk_size": 1024, "memory_attention_temperature": 0.07,
        "memory_reliability_epsilon": 1e-6, "memory_reliable_ratio_threshold": 0.01,
        "memory_confidence_threshold": 0.1, "memory_exclusion_extra_margin": 0,
        "boundary_width": 1, "route_tau_end": 0.2, "pair_alignment_policy": "strict",
        "tir_normalization": "dtype_range", "tir_percentile_low": 1.0,
        "tir_percentile_high": 99.0, "tir_percentile_scope": "per_image",
        "tir_channel_tolerance_code_values": 1, "tir_channel_tolerance_float": 1e-5,
    }
    checkpoint = tmp_path / "ema.pt"
    torch.save({"training_stage": "ema", "config": config, "student": model.state_dict(),
                "teacher": teacher.state_dict()}, checkpoint)
    hazy_dir, tir_dir = tmp_path / "hazy", tmp_path / "tir"
    hazy_dir.mkdir(); tir_dir.mkdir()
    Image.new("RGB", (47, 31), (80, 80, 80)).save(hazy_dir / "x.png")
    Image.new("L", (47, 31), 100).save(tir_dir / "x.png")

    for selected in ("student", "teacher"):
        out_dir = tmp_path / selected
        main(["--checkpoint", str(checkpoint), "--hazy_dir", str(hazy_dir), "--tir_dir", str(tir_dir),
              "--output_dir", str(out_dir), "--device", "cpu", "--ema_model", selected])
        assert Image.open(out_dir / "x.png").size == (47, 31)


def test_eval_pairs_case_insensitive_stems_across_extensions_and_ignores_non_images(tmp_path):
    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2)
    config = {
        "base_channels": 8, "router_hidden_channels": 8, "deform_num_samples": 4,
        "deform_max_offset": 2.0, "num_structure_renderers": 2, "memory_max_tokens": 16,
        "memory_topk": 2, "memory_query_chunk_size": 1024, "memory_attention_temperature": 0.07,
        "memory_reliability_epsilon": 1e-6, "memory_reliable_ratio_threshold": 0.01,
        "memory_confidence_threshold": 0.1, "memory_exclusion_extra_margin": 0,
        "boundary_width": 1, "route_tau_end": 0.2, "pair_alignment_policy": "strict",
        "tir_normalization": "dtype_range", "tir_percentile_low": 1.0,
        "tir_percentile_high": 99.0, "tir_percentile_scope": "per_image",
        "tir_channel_tolerance_code_values": 1, "tir_channel_tolerance_float": 1e-5,
    }
    checkpoint = tmp_path / "source.pt"
    torch.save({"training_stage": "source", "config": config, "model": model.state_dict()}, checkpoint)
    hazy_dir, tir_dir, out_dir = tmp_path / "hazy", tmp_path / "tir", tmp_path / "out"
    hazy_dir.mkdir(); tir_dir.mkdir()
    Image.new("RGB", (32, 32), (80, 80, 80)).save(hazy_dir / "X.jpg")
    Image.new("L", (32, 32), 100).save(tir_dir / "x.png")
    (hazy_dir / "notes.txt").write_text("not an image", encoding="utf-8")

    main(["--checkpoint", str(checkpoint), "--hazy_dir", str(hazy_dir), "--tir_dir", str(tir_dir),
          "--output_dir", str(out_dir), "--device", "cpu"])
    assert (out_dir / "X.png").is_file()


def test_eval_rejects_case_insensitive_duplicate_hazy_stems(tmp_path):
    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2)
    config = {
        "base_channels": 8, "router_hidden_channels": 8, "deform_num_samples": 4,
        "deform_max_offset": 2.0, "num_structure_renderers": 2, "memory_max_tokens": 16,
        "memory_topk": 2, "memory_query_chunk_size": 1024, "memory_attention_temperature": 0.07,
        "memory_reliability_epsilon": 1e-6, "memory_reliable_ratio_threshold": 0.01,
        "memory_confidence_threshold": 0.1, "memory_exclusion_extra_margin": 0,
        "boundary_width": 1, "route_tau_end": 0.2, "pair_alignment_policy": "strict",
        "tir_normalization": "dtype_range", "tir_percentile_low": 1.0,
        "tir_percentile_high": 99.0, "tir_percentile_scope": "per_image",
        "tir_channel_tolerance_code_values": 1, "tir_channel_tolerance_float": 1e-5,
    }
    checkpoint = tmp_path / "source.pt"
    torch.save({"training_stage": "source", "config": config, "model": model.state_dict()}, checkpoint)
    hazy_dir, tir_dir, out_dir = tmp_path / "hazy", tmp_path / "tir", tmp_path / "out"
    hazy_dir.mkdir(); tir_dir.mkdir()
    first_hazy, second_hazy = hazy_dir / "x.jpg", hazy_dir / "X.png"
    Image.new("RGB", (32, 32), (80, 80, 80)).save(first_hazy)
    Image.new("RGB", (32, 32), (80, 80, 80)).save(second_hazy)
    Image.new("L", (32, 32), 100).save(tir_dir / "x.tif")

    with pytest.raises(ValueError, match="ambiguous hazy stem=x") as error:
        main(["--checkpoint", str(checkpoint), "--hazy_dir", str(hazy_dir), "--tir_dir", str(tir_dir),
              "--output_dir", str(out_dir), "--device", "cpu"])
    assert str(first_hazy) in str(error.value)
    assert str(second_hazy) in str(error.value)


def test_eval_reports_missing_and_ambiguous_tir_stem_pairs(tmp_path):
    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2)
    config = {
        "base_channels": 8, "router_hidden_channels": 8, "deform_num_samples": 4,
        "deform_max_offset": 2.0, "num_structure_renderers": 2, "memory_max_tokens": 16,
        "memory_topk": 2, "memory_query_chunk_size": 1024, "memory_attention_temperature": 0.07,
        "memory_reliability_epsilon": 1e-6, "memory_reliable_ratio_threshold": 0.01,
        "memory_confidence_threshold": 0.1, "memory_exclusion_extra_margin": 0,
        "boundary_width": 1, "route_tau_end": 0.2, "pair_alignment_policy": "strict",
        "tir_normalization": "dtype_range", "tir_percentile_low": 1.0,
        "tir_percentile_high": 99.0, "tir_percentile_scope": "per_image",
        "tir_channel_tolerance_code_values": 1, "tir_channel_tolerance_float": 1e-5,
    }
    checkpoint = tmp_path / "source.pt"
    torch.save({"training_stage": "source", "config": config, "model": model.state_dict()}, checkpoint)
    hazy_dir, tir_dir, out_dir = tmp_path / "hazy", tmp_path / "tir", tmp_path / "out"
    hazy_dir.mkdir(); tir_dir.mkdir()
    Image.new("RGB", (32, 32), (80, 80, 80)).save(hazy_dir / "x.jpg")

    with pytest.raises(FileNotFoundError, match="stem=x"):
        main(["--checkpoint", str(checkpoint), "--hazy_dir", str(hazy_dir), "--tir_dir", str(tir_dir),
              "--output_dir", str(out_dir), "--device", "cpu"])
    Image.new("L", (32, 32), 100).save(tir_dir / "x.png")
    Image.new("L", (32, 32), 100).save(tir_dir / "X.tif")
    with pytest.raises(ValueError, match="ambiguous.*stem=x"):
        main(["--checkpoint", str(checkpoint), "--hazy_dir", str(hazy_dir), "--tir_dir", str(tir_dir),
              "--output_dir", str(out_dir), "--device", "cpu"])
