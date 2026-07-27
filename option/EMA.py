"""Pure configuration for real-domain EMA adaptation.

Source-model and source-anchor semantics are inherited from a checkpoint rather
than accepted as independent EMA command-line architecture options.
"""

import argparse
import json
import warnings
from pathlib import Path


LOSS_WEIGHT_NAMES = (
    "q_l1_weight", "q_gradient_weight", "q_ssim_weight",
    "rec_l1_weight", "rec_gradient_weight", "rec_ssim_weight",
    "boundary_l1_weight", "boundary_gradient_weight",
)
SOURCE_TRAINING_OBJECTIVE_KEYS = ("formal_training", *LOSS_WEIGHT_NAMES)
_LEGACY_SOURCE_OBJECTIVE_DEFAULTS = {
    "formal_training": False,
    "q_l1_weight": 1.0, "q_gradient_weight": 0.0, "q_ssim_weight": 0.0,
    "rec_l1_weight": 1.0, "rec_gradient_weight": 0.0, "rec_ssim_weight": 0.0,
    "boundary_l1_weight": 1.0, "boundary_gradient_weight": 0.0,
}
_FORMAL_TRAINING_LOSS_WEIGHTS = {
    "q_l1_weight": 1.0, "q_gradient_weight": 0.5, "q_ssim_weight": 0.5,
    "rec_l1_weight": 1.0, "rec_gradient_weight": 0.2, "rec_ssim_weight": 0.2,
    "boundary_l1_weight": 1.0, "boundary_gradient_weight": 0.5,
}

_INHERITED_SOURCE_KEYS = (
    "model_init_seed", "base_channels", "router_hidden_channels", "num_structure_renderers",
    "deform_num_samples", "deform_max_offset", "memory_max_tokens", "memory_topk",
    "memory_query_chunk_size", "memory_attention_temperature", "memory_reliability_epsilon",
    "memory_reliable_ratio_threshold", "memory_confidence_threshold",
    "memory_exclusion_extra_margin", "boundary_width", "route_tau_start", "route_tau_end",
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
    "reconstruction_ssim_window", "reconstruction_min_valid_support",
    "lambda_global", "lambda_fuse", "lambda_comp", "lambda_boundary", "lambda_router",
    "lambda_density", "lambda_route", "lambda_binary", *SOURCE_TRAINING_OBJECTIVE_KEYS,
)
_EMA_RUNTIME_KEYS = (
    "device", "real_data_dir", "source_anchor_data_dir", "real_batch_size",
    "source_anchor_batch_size", "num_workers", "source_checkpoint", "resume_checkpoint",
    "allow_ema_training_override", "exp_dir", "saved_model_dir", "saved_data_dir",
)
_EMA_TRAINING_KEYS = (
    "epochs", "learning_rate", "ema_decay", "ema_sigma_j", "ema_sigma_m", "ema_sigma_r",
    "ema_stability_min_weight", "lambda_ema_j", "lambda_ema_m", "lambda_ema_r",
    "lambda_anchor", "max_consecutive_failed_steps",
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
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--ema_sigma_j", type=float, default=0.1)
    parser.add_argument("--ema_sigma_m", type=float, default=0.1)
    parser.add_argument("--ema_sigma_r", type=float, default=0.1)
    parser.add_argument("--ema_stability_min_weight", type=float, default=0.05)
    parser.add_argument("--lambda_ema_j", type=float, default=1.0)
    parser.add_argument("--lambda_ema_m", type=float, default=1.0)
    parser.add_argument("--lambda_ema_r", type=float, default=1.0)
    parser.add_argument("--lambda_anchor", type=float, default=1.0)
    parser.add_argument("--max_consecutive_failed_steps", type=int, default=20)
    parser.add_argument("--allow_ema_training_override", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--exp_dir", default="experiment")
    parser.add_argument("--saved_model_dir", default="")
    parser.add_argument("--saved_data_dir", default="")
    return parser


def _normalise_checkpoint_config(checkpoint_config):
    if not isinstance(checkpoint_config, dict):
        raise TypeError("checkpoint config must be a dictionary")
    config = {key: value for key, value in checkpoint_config.items() if not key.startswith("_")}
    if "source_anchor_data_dir" in config:
        source_anchor_data_dir = config.pop("source_anchor_data_dir")
        config.pop("train_data_dir", None)
    else:
        source_anchor_data_dir = config.pop("train_data_dir", "")
    config["source_anchor_data_dir"] = source_anchor_data_dir
    missing = [key for key in SOURCE_TRAINING_OBJECTIVE_KEYS if key not in config]
    if missing:
        warnings.warn(
            "legacy checkpoint lacks source training objective fields; "
            "using historical L1 defaults for: " + ", ".join(missing),
            RuntimeWarning,
            stacklevel=3,
        )
        for key in missing:
            config[key] = _LEGACY_SOURCE_OBJECTIVE_DEFAULTS[key]
        if any(key in LOSS_WEIGHT_NAMES for key in missing):
            config["formal_training"] = False
    # Saved values must never be treated as omitted CLI weights and rewritten
    # by the formal preset during validation.
    config["_explicit_training_objective_keys"] = list(LOSS_WEIGHT_NAMES)
    return config


def resolve_ema_config(current, checkpoint_config, *, allow_training_override, is_resume=False):
    """Merge EMA runtime controls with immutable source checkpoint semantics."""
    current_config = vars(current) if isinstance(current, argparse.Namespace) else dict(current)
    resolved = _normalise_checkpoint_config(checkpoint_config)
    source_anchor_from_checkpoint = resolved["source_anchor_data_dir"]

    for key in _EMA_RUNTIME_KEYS:
        if key == "source_anchor_data_dir":
            resolved[key] = current_config[key] or source_anchor_from_checkpoint
        else:
            resolved[key] = current_config[key]

    differences = {}
    for key in _EMA_TRAINING_KEYS:
        stored = resolved.get(key)
        requested = current_config[key]
        if not is_resume or allow_training_override or stored is None:
            resolved[key] = requested
            if is_resume and stored is not None and stored != requested:
                differences[key] = (stored, requested)
        else:
            resolved[key] = stored
    return resolved, differences


def _validate_inherited_source_config(args):
    missing = [key for key in _INHERITED_SOURCE_KEYS if not hasattr(args, key)]
    if missing:
        raise ValueError("checkpoint lacks required source configuration: " + ", ".join(missing))
    for name in ("route_tau_start", "route_tau_end", "memory_attention_temperature", "q_temperature", "density_smooth_l1_beta"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be > 0")
    if args.route_hard_start_step < 0 or args.counterfactual_start_step > args.route_loss_start_step:
        raise ValueError("invalid route/counterfactual schedule")
    if args.counterfactual_chunk_size < 1 or args.deform_num_samples < 1 or args.deform_max_offset < 0:
        raise ValueError("invalid counterfactual/deform configuration")
    if args.train_size < 1 or args.num_structure_renderers < 1 or args.memory_max_tokens < 1 or args.memory_query_chunk_size < 1:
        raise ValueError("invalid source renderer, memory, or train-size configuration")
    if args.memory_exclusion_extra_margin < 0 or not 2 <= args.memory_topk <= args.memory_max_tokens:
        raise ValueError("invalid source memory configuration")
    for name in ("memory_reliable_ratio_threshold", "memory_confidence_threshold"):
        if not 0 <= getattr(args, name) <= 1:
            raise ValueError(f"{name} must be in [0,1]")
    if args.boundary_width < 1 or args.max_consecutive_empty_omega_steps < 1:
        raise ValueError("invalid source boundary or empty-Omega configuration")
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
        explicit = set(getattr(args, "_explicit_training_objective_keys", ()))
        for name, value in _FORMAL_TRAINING_LOSS_WEIGHTS.items():
            if name not in explicit:
                setattr(args, name, value)
        invalid = [name for name in LOSS_WEIGHT_NAMES if getattr(args, name) <= 0]
        if invalid:
            raise ValueError("formal_training requires positive source loss weights: " + ", ".join(invalid))
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
    if not 0 <= args.ema_decay < 1:
        raise ValueError("ema_decay must be in [0,1)")
    if min(args.ema_sigma_j, args.ema_sigma_m, args.ema_sigma_r) <= 0:
        raise ValueError("EMA sigmas must be > 0")
    if not 0 <= args.ema_stability_min_weight <= 1:
        raise ValueError("ema_stability_min_weight must be in [0,1]")
    if args.real_batch_size < 1 or args.source_anchor_batch_size < 1 or args.num_workers < 0:
        raise ValueError("EMA batch sizes must be positive and num_workers must be non-negative")
    if args.epochs < 1 or args.learning_rate <= 0 or args.max_consecutive_failed_steps < 1:
        raise ValueError("EMA epochs, learning_rate, and failure limit must be positive")
    return args


def build_ema_checkpoint_config(model_config, args):
    """Persist inherited source semantics plus the actual EMA-stage settings."""
    config = _normalise_checkpoint_config(model_config)
    config.pop("train_data_dir", None)
    config.pop("_explicit_training_objective_keys", None)
    current = persisted_config_from_args(args)
    for key in (*_EMA_RUNTIME_KEYS, *_EMA_TRAINING_KEYS):
        config[key] = current[key]
    config.pop("resume_checkpoint", None)
    return config


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
