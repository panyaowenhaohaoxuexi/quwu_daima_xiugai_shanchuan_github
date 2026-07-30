import pytest

from option.UDA import build_parser, resolve_uda_config, validate_config


def test_uda_parser_exposes_two_stages_and_target_style_defaults():
    args = build_parser().parse_args([])

    assert args.stage == "source_style"
    assert args.style_probability == 0.5
    assert args.style_beta_min == 0.0
    assert args.style_beta_max == 0.6
    assert args.route_consistency_warmup_steps == 1000
    assert args.route_consistency_ramp_steps == 1000


def test_uda_config_inherits_model_semantics_from_source_checkpoint():
    raw = build_parser().parse_args([])
    checkpoint = {
        "base_channels": 16, "router_hidden_channels": 8, "num_structure_renderers": 2,
        "deform_num_samples": 4, "deform_max_offset": 2.0, "memory_max_tokens": 256,
        "memory_topk": 8, "memory_query_chunk_size": 1024, "memory_attention_temperature": 0.07,
        "memory_reliability_epsilon": 1e-6, "memory_reliable_ratio_threshold": 0.01,
        "memory_confidence_threshold": 0.1, "decoder_num_heads": 4, "decoder_depth": 1,
        "decoder_window_size": 7, "decoder_window_chunk_size": 128, "decoder_mlp_ratio": 4.0,
        "decoder_attention_dropout": 0.0, "decoder_projection_dropout": 0.0, "decoder_ffn_dropout": 0.0,
        "model_init_seed": 1, "route_tau_start": 1.0, "route_tau_end": 0.2,
        "route_temperature_anneal_steps": 3, "route_teacher_anneal_steps": 3, "train_size": 256,
        "density_gt_semantics": "density", "density_map_normalization": "dtype_range",
        "density_fixed_min": None, "density_fixed_max": None, "density_calibrated_min": None,
        "density_calibrated_max": None, "tir_normalization": "dtype_range", "tir_fixed_min": None,
        "tir_fixed_max": None, "tir_percentile_low": 1.0, "tir_percentile_high": 99.0,
        "tir_percentile_scope": "per_image", "tir_dataset_percentile_low_value": None,
        "tir_dataset_percentile_high_value": None, "tir_channel_tolerance_code_values": 1,
        "tir_channel_tolerance_float": 1e-5, "pair_alignment_policy": "strict",
        "density_smooth_l1_beta": 0.1, "lambda_density": 1.0, "lambda_route": 1.0,
        "lambda_global": 1.0, "lambda_fuse": 1.0, "lambda_comp": 1.0, "lambda_boundary": 1.0,
        "global_l1_weight": 0.8, "global_ssim_weight": 0.2, "global_contrast_weight": 0.05,
        "region_l1_weight": 1.0, "region_gradient_weight": 0.2, "region_ssim_weight": 0.2,
        "reconstruction_ssim_window": 7, "reconstruction_min_valid_support": 4,
    }
    resolved = resolve_uda_config(raw, checkpoint)
    assert resolved["base_channels"] == 16
    assert resolved["style_probability"] == 0.5


@pytest.mark.parametrize("argv, message", [
    (["--style_probability", "1.1"], "style_probability"),
    (["--style_beta_min", "0.8", "--style_beta_max", "0.2"], "style beta"),
    (["--style_min_gain", "1.5", "--style_max_gain", "1.0"], "style gain"),
    (["--route_consistency_warmup_steps", "-1"], "route consistency"),
])
def test_uda_rejects_invalid_style_and_route_schedule_values(argv, message):
    args = build_parser().parse_args(argv)

    with pytest.raises(ValueError, match=message):
        validate_config(args)
