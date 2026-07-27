"""Pure configuration for synthetic-domain fog-routed training."""

import argparse
import json
from pathlib import Path


LOSS_WEIGHT_NAMES = (
    "q_l1_weight", "q_gradient_weight", "q_ssim_weight",
    "rec_l1_weight", "rec_gradient_weight", "rec_ssim_weight",
    "boundary_l1_weight", "boundary_gradient_weight",
)
TRAINING_OBJECTIVE_KEYS = ("formal_training", *LOSS_WEIGHT_NAMES)
FORMAL_TRAINING_LOSS_WEIGHTS = {
    "q_l1_weight": 1.0, "q_gradient_weight": 0.5, "q_ssim_weight": 0.5,
    "rec_l1_weight": 1.0, "rec_gradient_weight": 0.2, "rec_ssim_weight": 0.2,
    "boundary_l1_weight": 1.0, "boundary_gradient_weight": 0.5,
}


def _mark_explicit_training_objective(namespace, name):
    explicit = list(getattr(namespace, "_explicit_training_objective_keys", ()))
    if name not in explicit:
        explicit.append(name)
    setattr(namespace, "_explicit_training_objective_keys", explicit)


class _RecordExplicitTrainingObjective(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        _mark_explicit_training_objective(namespace, self.dest)


class _RecordExplicitFormalTraining(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, True)
        _mark_explicit_training_objective(namespace, self.dest)


def _add_loss_weight_argument(parser, name, default):
    parser.add_argument(f"--{name}", type=float, default=default, action=_RecordExplicitTrainingObjective)


def persisted_config_from_args(args):
    """Return config/checkpoint fields without parser-only bookkeeping."""
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


def add_model_arguments(parser):
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model_init_seed", type=int, default=20260724)
    parser.add_argument("--base_channels", type=int, default=16)
    parser.add_argument("--router_hidden_channels", type=int, default=8)
    parser.add_argument("--num_structure_renderers", type=int, default=2)
    parser.add_argument("--deform_num_samples", type=int, default=4)
    parser.add_argument("--deform_max_offset", type=float, default=2.0)
    parser.add_argument("--memory_max_tokens", type=int, default=256)
    parser.add_argument("--memory_topk", type=int, default=8)
    parser.add_argument("--memory_query_chunk_size", type=int, default=1024)
    parser.add_argument("--memory_attention_temperature", type=float, default=0.07)
    parser.add_argument("--memory_reliability_epsilon", type=float, default=1e-6)
    parser.add_argument("--memory_reliable_ratio_threshold", type=float, default=0.01)
    parser.add_argument("--memory_confidence_threshold", type=float, default=0.1)
    parser.add_argument("--memory_exclusion_extra_margin", type=int, default=0)
    parser.add_argument("--boundary_width", type=int, default=1)
    parser.add_argument("--route_tau_start", type=float, default=1.0)
    parser.add_argument("--route_tau_end", type=float, default=0.2)
    parser.add_argument("--route_hard_start_step", type=int, default=10000)


def add_synthetic_data_arguments(parser):
    parser.add_argument("--train_data_dir", default="")
    parser.add_argument("--train_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--density_gt_semantics", choices=("transmission", "density"), default="transmission")
    parser.add_argument("--density_map_normalization", choices=("dtype_range", "fixed_range", "dataset_calibrated_range"), default="dtype_range")
    parser.add_argument("--density_fixed_min", type=float)
    parser.add_argument("--density_fixed_max", type=float)
    parser.add_argument("--density_calibrated_min", type=float)
    parser.add_argument("--density_calibrated_max", type=float)
    parser.add_argument("--tir_normalization", choices=("dtype_range", "fixed_range", "percentile"), default="dtype_range")
    parser.add_argument("--tir_fixed_min", type=float)
    parser.add_argument("--tir_fixed_max", type=float)
    parser.add_argument("--tir_percentile_low", type=float, default=1.0)
    parser.add_argument("--tir_percentile_high", type=float, default=99.0)
    parser.add_argument("--tir_percentile_scope", choices=("per_image", "dataset"), default="per_image")
    parser.add_argument("--tir_dataset_percentile_low_value", type=float)
    parser.add_argument("--tir_dataset_percentile_high_value", type=float)
    parser.add_argument("--tir_channel_tolerance_code_values", type=int, default=1)
    parser.add_argument("--tir_channel_tolerance_float", type=float, default=1e-5)
    parser.add_argument("--pair_alignment_policy", choices=("strict", "resize_tir_to_rgb"), default="strict")


def add_source_training_arguments(parser):
    parser.add_argument("--counterfactual_chunk_size", type=int, default=4)
    parser.add_argument("--counterfactual_start_step", type=int, default=1000)
    parser.add_argument("--route_loss_start_step", type=int, default=1000)
    parser.add_argument("--route_loss_warmup_steps", type=int, default=1000)
    parser.add_argument("--binary_loss_start_step", type=int, default=5000)
    parser.add_argument("--binary_loss_warmup_steps", type=int, default=1000)
    parser.add_argument("--omega_regions_per_image", type=int, default=6)
    parser.add_argument("--omega_min_area", type=int, default=16)
    parser.add_argument("--omega_max_area", type=int, default=256)
    parser.add_argument("--max_consecutive_empty_omega_steps", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--resume_checkpoint", default="")


def add_source_loss_arguments(parser):
    parser.add_argument("--q_temperature", type=float, default=0.1)
    parser.add_argument("--q_window_size", type=int, default=7)
    parser.add_argument("--q_min_valid_support", type=int, default=4)
    parser.add_argument("--formal_training", action=_RecordExplicitFormalTraining, nargs=0, default=False)
    _add_loss_weight_argument(parser, "q_l1_weight", 1.0)
    _add_loss_weight_argument(parser, "q_gradient_weight", 0.0)
    _add_loss_weight_argument(parser, "q_ssim_weight", 0.0)
    parser.add_argument("--density_smooth_l1_beta", type=float, default=0.1)
    _add_loss_weight_argument(parser, "rec_l1_weight", 1.0)
    _add_loss_weight_argument(parser, "rec_gradient_weight", 0.0)
    _add_loss_weight_argument(parser, "rec_ssim_weight", 0.0)
    _add_loss_weight_argument(parser, "boundary_l1_weight", 1.0)
    _add_loss_weight_argument(parser, "boundary_gradient_weight", 0.0)
    parser.add_argument("--reconstruction_ssim_window", type=int, default=7)
    parser.add_argument("--reconstruction_min_valid_support", type=int, default=4)
    for name in ("global", "fuse", "comp", "boundary", "router", "density", "route", "binary"):
        parser.add_argument(f"--lambda_{name}", type=float, default=1.0)


def add_output_arguments(parser):
    parser.add_argument("--exp_dir", default="experiment")
    parser.add_argument("--saved_model_dir", default="")
    parser.add_argument("--saved_data_dir", default="")


def _validate_preprocessing(args):
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


def _validate_model_and_source_flow(args):
    for name in ("route_tau_start", "route_tau_end", "memory_attention_temperature", "q_temperature", "density_smooth_l1_beta"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be > 0")
    if args.route_hard_start_step < 0 or args.counterfactual_start_step > args.route_loss_start_step:
        raise ValueError("invalid route/counterfactual schedule")
    if args.counterfactual_chunk_size < 1 or args.deform_num_samples < 1 or args.deform_max_offset < 0:
        raise ValueError("invalid counterfactual/deform configuration")
    if args.train_size < 1 or args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("train_size/batch_size must be positive and num_workers must be non-negative")
    if args.num_structure_renderers < 1 or args.memory_max_tokens < 1 or args.memory_query_chunk_size < 1:
        raise ValueError("invalid renderer or memory configuration")
    if args.memory_exclusion_extra_margin < 0:
        raise ValueError("memory_exclusion_extra_margin must be >= 0")
    if not 2 <= args.memory_topk <= args.memory_max_tokens:
        raise ValueError("require 2 <= memory_topk <= memory_max_tokens")
    for name in ("memory_reliable_ratio_threshold", "memory_confidence_threshold"):
        if not 0 <= getattr(args, name) <= 1:
            raise ValueError(f"{name} must be in [0,1]")
    if args.boundary_width < 1 or args.max_consecutive_empty_omega_steps < 1:
        raise ValueError("boundary_width and consecutive-step limits must be >= 1")
    if not (4 <= args.omega_regions_per_image <= 8) or not (0 < args.omega_min_area <= args.omega_max_area):
        raise ValueError("invalid Omega configuration")


def validate_source_loss_arguments(args):
    if args.q_window_size <= 0 or args.q_window_size % 2 == 0:
        raise ValueError("q_window_size must be positive and odd")
    if args.q_min_valid_support < 1:
        raise ValueError("q_min_valid_support must be >= 1")
    negative = [name for name in LOSS_WEIGHT_NAMES if getattr(args, name) < 0]
    if negative:
        raise ValueError("loss weights must be non-negative: " + ", ".join(negative))
    if args.formal_training:
        explicit = set(getattr(args, "_explicit_training_objective_keys", ()))
        if not getattr(args, "_resume_checkpoint_semantics", False):
            for name, value in FORMAL_TRAINING_LOSS_WEIGHTS.items():
                if name not in explicit:
                    setattr(args, name, value)
        invalid = [name for name in LOSS_WEIGHT_NAMES if getattr(args, name) <= 0]
        if invalid:
            raise ValueError("formal_training cannot use smoke default loss weights; all formal loss weights must be > 0: " + ", ".join(invalid))
    if args.reconstruction_ssim_window <= 0 or args.reconstruction_ssim_window % 2 == 0:
        raise ValueError("reconstruction_ssim_window must be positive and odd")
    if args.reconstruction_min_valid_support < 1:
        raise ValueError("reconstruction_min_valid_support must be >= 1")
    if any(getattr(args, name) < 0 for name in vars(args) if name.startswith("lambda_")):
        raise ValueError("lambda coefficients must be non-negative")


def build_parser():
    parser = argparse.ArgumentParser("fog-routed-source")
    add_model_arguments(parser)
    add_synthetic_data_arguments(parser)
    add_source_training_arguments(parser)
    add_source_loss_arguments(parser)
    add_output_arguments(parser)
    return parser


def validate_config(args):
    _validate_model_and_source_flow(args)
    _validate_preprocessing(args)
    validate_source_loss_arguments(args)
    if args.epochs < 1 or args.learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")
    return args


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
