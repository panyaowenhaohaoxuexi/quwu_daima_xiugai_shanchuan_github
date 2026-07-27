"""Pure configuration for real-domain EMA adaptation."""

import argparse
import json
from pathlib import Path


LOSS_WEIGHT_NAMES = (
    "q_l1_weight", "q_gradient_weight", "q_ssim_weight",
    "rec_l1_weight", "rec_gradient_weight", "rec_ssim_weight",
    "boundary_l1_weight", "boundary_gradient_weight",
)

INHERITED_SOURCE_KEYS = (
    "model_init_seed", "base_channels", "router_hidden_channels", "num_structure_renderers",
    "deform_num_samples", "deform_max_offset", "memory_max_tokens", "memory_topk",
    "memory_query_chunk_size", "memory_attention_temperature", "memory_reliability_epsilon",
    "memory_reliable_ratio_threshold", "memory_confidence_threshold",
    "memory_exclusion_extra_margin", "boundary_width", "route_tau_start", "route_tau_end",
    "decoder_num_heads", "decoder_depth", "decoder_window_size", "decoder_window_chunk_size",
    "decoder_mlp_ratio", "decoder_attention_dropout", "decoder_projection_dropout", "decoder_ffn_dropout",
    "route_hard_start_step", "train_size", "density_gt_semantics", "density_map_normalization",
    "density_fixed_min", "density_fixed_max", "density_calibrated_min",
    "density_calibrated_max", "tir_normalization", "tir_fixed_min", "tir_fixed_max",
    "tir_percentile_low", "tir_percentile_high", "tir_percentile_scope",
    "tir_dataset_percentile_low_value", "tir_dataset_percentile_high_value",
    "tir_channel_tolerance_code_values", "tir_channel_tolerance_float",
    "pair_alignment_policy", "counterfactual_chunk_size", "counterfactual_start_step",
    "route_loss_start_step", "route_loss_warmup_steps", "binary_loss_start_step",
    "binary_loss_warmup_steps", "omega_regions_per_image", "omega_min_area",
    "omega_max_area", "max_consecutive_empty_omega_steps", "q_temperature",
    "q_window_size", "q_min_valid_support", "density_smooth_l1_beta",
    "reconstruction_ssim_window", "reconstruction_min_valid_support", "formal_training",
    "lambda_global", "lambda_fuse", "lambda_comp", "lambda_boundary", "lambda_router",
    "lambda_density", "lambda_route", "lambda_binary", *LOSS_WEIGHT_NAMES,
)
EMA_PARSER_KEYS = (
    "device", "real_data_dir", "source_anchor_data_dir", "source_checkpoint", "resume_checkpoint",
    "real_batch_size", "source_anchor_batch_size", "num_workers", "epochs", "learning_rate",
    "ema_decay", "ema_sigma_j", "ema_sigma_m", "ema_sigma_r", "ema_stability_min_weight",
    "lambda_ema_j", "lambda_ema_m", "lambda_ema_r", "lambda_anchor",
    "exp_dir", "saved_model_dir", "saved_data_dir",
)
EMA_RUNTIME_KEYS = (
    "device", "real_data_dir", "source_anchor_data_dir", "source_checkpoint", "resume_checkpoint",
    "real_batch_size", "source_anchor_batch_size", "num_workers", "epochs", "learning_rate",
    "exp_dir", "saved_model_dir", "saved_data_dir",
)
EMA_SEMANTIC_KEYS = tuple(key for key in EMA_PARSER_KEYS if key not in EMA_RUNTIME_KEYS)
EMA_CHECKPOINT_CONFIG_KEYS = tuple(
    key for key in EMA_PARSER_KEYS if key not in {"source_checkpoint", "resume_checkpoint"}
)


def persisted_config_from_args(args):
    return {key: value for key, value in vars(args).items() if not key.startswith("_")}


def tir_normalization_config_from_args(args):
    return {
        "normalization": args.tir_normalization,
        "fixed_min": args.tir_fixed_min,
        "fixed_max": args.tir_fixed_max,
        "percentile_low": args.tir_percentile_low,
        "percentile_high": args.tir_percentile_high,
        "percentile_scope": args.tir_percentile_scope,
        "dataset_percentile_low_value": args.tir_dataset_percentile_low_value,
        "dataset_percentile_high_value": args.tir_dataset_percentile_high_value,
        "channel_tolerance_code_values": args.tir_channel_tolerance_code_values,
        "channel_tolerance_float": args.tir_channel_tolerance_float,
    }


def build_parser():
    parser = argparse.ArgumentParser("fog-routed-ema")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--real_data_dir", default="")
    parser.add_argument("--source_anchor_data_dir", default="")
    parser.add_argument("--source_checkpoint", default="")
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--real_batch_size", type=int, default=1)
    parser.add_argument("--source_anchor_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--ema_sigma_j", type=float, default=0.1)
    parser.add_argument("--ema_sigma_m", type=float, default=0.1)
    parser.add_argument("--ema_sigma_r", type=float, default=0.1)
    parser.add_argument("--ema_stability_min_weight", type=float, default=0.05)
    parser.add_argument("--lambda_ema_j", type=float, default=1.0)
    parser.add_argument("--lambda_ema_m", type=float, default=1.0)
    parser.add_argument("--lambda_ema_r", type=float, default=1.0)
    parser.add_argument("--lambda_anchor", type=float, default=1.0)
    parser.add_argument("--exp_dir", default="experiment")
    parser.add_argument("--saved_model_dir", default="")
    parser.add_argument("--saved_data_dir", default="")
    return parser


def _inherited_source_config(checkpoint_config):
    if not isinstance(checkpoint_config, dict):
        raise TypeError("checkpoint config must be a dictionary")
    missing = [key for key in INHERITED_SOURCE_KEYS if key not in checkpoint_config]
    if missing:
        raise ValueError("checkpoint lacks required Source configuration: " + ", ".join(missing))
    return {key: checkpoint_config[key] for key in INHERITED_SOURCE_KEYS}


def resolve_ema_config(raw_args, checkpoint_config, *, resume=False):
    """Merge checkpoint semantics with only the current run's runtime settings."""
    raw_config = vars(raw_args) if isinstance(raw_args, argparse.Namespace) else dict(raw_args)
    inherited = _inherited_source_config(checkpoint_config)
    if resume:
        missing = [key for key in EMA_SEMANTIC_KEYS if key not in checkpoint_config]
        if missing:
            raise ValueError("EMA checkpoint lacks required adaptation configuration: " + ", ".join(missing))
        semantics = {key: checkpoint_config[key] for key in EMA_SEMANTIC_KEYS}
    else:
        semantics = {key: raw_config[key] for key in EMA_SEMANTIC_KEYS}
    return {**inherited, **semantics, **{key: raw_config[key] for key in EMA_RUNTIME_KEYS}}


def _validate_inherited_source_config(args):
    missing = [key for key in INHERITED_SOURCE_KEYS if not hasattr(args, key)]
    if missing:
        raise ValueError("EMA config lacks required Source configuration: " + ", ".join(missing))
    for name in ("route_tau_start", "route_tau_end", "memory_attention_temperature", "q_temperature", "density_smooth_l1_beta"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be > 0")
    if args.route_hard_start_step < 0 or args.counterfactual_start_step > args.route_loss_start_step:
        raise ValueError("invalid route/counterfactual schedule")
    if args.counterfactual_chunk_size < 1 or args.deform_num_samples < 1 or args.deform_max_offset < 0:
        raise ValueError("invalid counterfactual/deform configuration")
    if args.train_size < 1 or args.num_structure_renderers < 1 or args.memory_max_tokens < 1 or args.memory_query_chunk_size < 1:
        raise ValueError("invalid Source renderer, memory, or train-size configuration")
    if args.memory_exclusion_extra_margin < 0 or not 2 <= args.memory_topk <= args.memory_max_tokens:
        raise ValueError("invalid Source memory configuration")
    if args.base_channels <= 0 or args.decoder_num_heads <= 0:
        raise ValueError("base_channels and decoder_num_heads must be > 0")
    widths = (args.base_channels, args.base_channels * 2, args.base_channels * 3, args.base_channels * 4)
    if any(width % args.decoder_num_heads for width in widths):
        raise ValueError("all decoder scale widths must be divisible by decoder_num_heads; "
                         f"decoder_num_heads={args.decoder_num_heads}, widths={widths}")
    for name in ("decoder_depth", "decoder_window_size", "decoder_window_chunk_size", "decoder_mlp_ratio"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be > 0, received {getattr(args, name)}")
    for name in ("decoder_attention_dropout", "decoder_projection_dropout", "decoder_ffn_dropout"):
        value = getattr(args, name)
        if not 0.0 <= value < 1.0:
            raise ValueError(f"{name} must be in [0, 1), received {value}")
    for name in ("memory_reliable_ratio_threshold", "memory_confidence_threshold"):
        if not 0 <= getattr(args, name) <= 1:
            raise ValueError(f"{name} must be in [0,1]")
    if args.boundary_width < 1 or args.max_consecutive_empty_omega_steps < 1:
        raise ValueError("invalid Source boundary or empty-Omega configuration")
    if not (4 <= args.omega_regions_per_image <= 8) or not (0 < args.omega_min_area <= args.omega_max_area):
        raise ValueError("invalid Omega configuration")
    if args.q_window_size <= 0 or args.q_window_size % 2 == 0 or args.q_min_valid_support < 1:
        raise ValueError("invalid q supervision configuration")
    if args.reconstruction_ssim_window <= 0 or args.reconstruction_ssim_window % 2 == 0:
        raise ValueError("invalid reconstruction SSIM configuration")
    if args.reconstruction_min_valid_support < 1:
        raise ValueError("reconstruction_min_valid_support must be >= 1")
    negative = [name for name in LOSS_WEIGHT_NAMES if getattr(args, name) < 0]
    if negative:
        raise ValueError("loss weights must be non-negative: " + ", ".join(negative))
    if args.formal_training:
        invalid = [name for name in LOSS_WEIGHT_NAMES if getattr(args, name) <= 0]
        if invalid:
            raise ValueError("formal_training requires positive Source loss weights: " + ", ".join(invalid))
    if any(getattr(args, name) < 0 for name in vars(args) if name.startswith("lambda_")):
        raise ValueError("lambda coefficients must be non-negative")
    if args.tir_normalization == "fixed_range" and not (args.tir_fixed_max > args.tir_fixed_min):
        raise ValueError("TIR fixed max must exceed min")
    if not (0 <= args.tir_percentile_low < args.tir_percentile_high <= 100):
        raise ValueError("invalid TIR percentiles")
    if args.tir_normalization == "percentile" and args.tir_percentile_scope == "dataset":
        if args.tir_dataset_percentile_low_value is None or args.tir_dataset_percentile_high_value is None:
            raise ValueError("dataset TIR percentile scope requires calibrated low/high values")
        if not args.tir_dataset_percentile_high_value > args.tir_dataset_percentile_low_value:
            raise ValueError("dataset TIR percentile high value must exceed low value")
    if args.density_map_normalization == "fixed_range" and not (args.density_fixed_max > args.density_fixed_min):
        raise ValueError("density fixed max must exceed min")
    if args.density_map_normalization == "dataset_calibrated_range" and not (args.density_calibrated_max > args.density_calibrated_min):
        raise ValueError("density calibrated max must exceed min")


def validate_config(args):
    _validate_inherited_source_config(args)
    if not args.real_data_dir:
        raise ValueError("real_data_dir must be provided for EMA adaptation")
    if not args.source_anchor_data_dir:
        raise ValueError("source_anchor_data_dir must be provided for EMA adaptation")
    if not 0 <= args.ema_decay < 1:
        raise ValueError("ema_decay must be in [0,1)")
    if min(args.ema_sigma_j, args.ema_sigma_m, args.ema_sigma_r) <= 0:
        raise ValueError("EMA sigmas must be > 0")
    if not 0 <= args.ema_stability_min_weight <= 1:
        raise ValueError("ema_stability_min_weight must be in [0,1]")
    if args.real_batch_size < 1 or args.source_anchor_batch_size < 1 or args.num_workers < 0:
        raise ValueError("EMA batch sizes must be positive and num_workers must be non-negative")
    if args.epochs < 1 or args.learning_rate <= 0:
        raise ValueError("EMA epochs and learning_rate must be positive")
    return args


def build_ema_checkpoint_config(model_config, args):
    """Persist only EMA-required Source semantics and active EMA settings."""
    inherited = _inherited_source_config(model_config)
    current = persisted_config_from_args(args)
    return {
        **inherited,
        **{key: current[key] for key in EMA_CHECKPOINT_CONFIG_KEYS},
    }


def prepare_experiment_dirs(args):
    for value in (args.exp_dir, args.saved_model_dir, args.saved_data_dir):
        if value:
            Path(value).mkdir(parents=True, exist_ok=True)


def save_config(args):
    if not args.exp_dir:
        return
    Path(args.exp_dir).mkdir(parents=True, exist_ok=True)
    with (Path(args.exp_dir) / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(persisted_config_from_args(args), handle, indent=2, sort_keys=True)
