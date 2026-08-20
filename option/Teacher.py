"""Configuration for V2 physical-mask Source training."""

import argparse
import json
from pathlib import Path

from utils.model_config_validation import validate_model_config_values


LOCAL_SOURCE_TRAIN_DIR = r"/root/autodl-tmp/1_FLIR/train"
LOCAL_VALIDATION_DATA_DIR = r"/root/autodl-tmp/1_FLIR/test"
LOCAL_TEACHER_OUTPUT_DIR = r"/root/autodl-tmp/train_data_model/1_Teacher_train"
# Optional fixed real-domain probe.  Leave all three empty to disable it.
LOCAL_SOURCE_PROBE_HAZY = r""
LOCAL_SOURCE_PROBE_TIR = r""
LOCAL_SOURCE_PROBE_OUTPUT_DIR = r""


def persisted_config_from_args(args):
    return {key: value for key, value in vars(args).items() if not key.startswith("_")}


def tir_normalization_config_from_args(args):
    return {"normalization": args.tir_normalization, "fixed_min": args.tir_fixed_min,
            "fixed_max": args.tir_fixed_max, "percentile_low": args.tir_percentile_low,
            "percentile_high": args.tir_percentile_high, "percentile_scope": args.tir_percentile_scope,
            "dataset_percentile_low_value": args.tir_dataset_percentile_low_value,
            "dataset_percentile_high_value": args.tir_dataset_percentile_high_value,
            "channel_tolerance_code_values": args.tir_channel_tolerance_code_values,
            "channel_tolerance_float": args.tir_channel_tolerance_float}


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
    parser.add_argument("--decoder_num_heads", type=int, default=4)
    parser.add_argument("--decoder_depth", type=int, default=1)
    parser.add_argument("--decoder_window_size", type=int, default=7)
    parser.add_argument("--decoder_window_chunk_size", type=int, default=128)
    parser.add_argument("--decoder_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--decoder_attention_dropout", type=float, default=0.0)
    parser.add_argument("--decoder_projection_dropout", type=float, default=0.0)
    parser.add_argument("--decoder_ffn_dropout", type=float, default=0.0)
    parser.add_argument("--route_tau_start", type=float, default=1.0)
    parser.add_argument("--route_tau_end", type=float, default=0.2)
    parser.add_argument("--route_temperature_anneal_steps", type=int, default=3000)
    parser.add_argument("--route_teacher_anneal_steps", type=int, default=3000)


def add_synthetic_data_arguments(parser):
    parser.add_argument("--train_data_dir", default=LOCAL_SOURCE_TRAIN_DIR)
    parser.add_argument("--validation_data_dir", default=LOCAL_VALIDATION_DATA_DIR)
    parser.add_argument("--train_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--validation_batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--density_gt_semantics", choices=("density",), default="density")
    parser.add_argument("--density_map_normalization", choices=("dtype_range", "fixed_range", "dataset_calibrated_range"), default="dtype_range")
    parser.add_argument("--density_fixed_min", type=float); parser.add_argument("--density_fixed_max", type=float)
    parser.add_argument("--density_calibrated_min", type=float); parser.add_argument("--density_calibrated_max", type=float)
    parser.add_argument("--tir_normalization", choices=("dtype_range", "fixed_range", "percentile"), default="dtype_range")
    parser.add_argument("--tir_fixed_min", type=float); parser.add_argument("--tir_fixed_max", type=float)
    parser.add_argument("--tir_percentile_low", type=float, default=1.0); parser.add_argument("--tir_percentile_high", type=float, default=99.0)
    parser.add_argument("--tir_percentile_scope", choices=("per_image", "dataset"), default="per_image")
    parser.add_argument("--tir_dataset_percentile_low_value", type=float); parser.add_argument("--tir_dataset_percentile_high_value", type=float)
    parser.add_argument("--tir_channel_tolerance_code_values", type=int, default=1); parser.add_argument("--tir_channel_tolerance_float", type=float, default=1e-5)
    parser.add_argument("--pair_alignment_policy", choices=("strict", "resize_tir_to_rgb"), default="strict")


def add_training_arguments(parser):
    parser.add_argument("--epochs", type=int, default=30); parser.add_argument("--iters_per_epoch", type=int, default=1000)
    parser.add_argument("--start_lr", "--learning_rate", dest="start_lr", type=float, default=1e-4); parser.add_argument("--end_lr", type=float, default=1e-6)
    parser.add_argument("--no_lr_sche", action="store_true"); parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--density_smooth_l1_beta", type=float, default=0.1)
    parser.add_argument("--lambda_density", type=float, default=1.0); parser.add_argument("--lambda_route", type=float, default=1.0)
    parser.add_argument("--lambda_global", type=float, default=1.0); parser.add_argument("--lambda_fuse", type=float, default=1.0)
    parser.add_argument("--lambda_comp", type=float, default=1.0); parser.add_argument("--lambda_boundary", type=float, default=1.0)
    parser.add_argument("--global_l1_weight", type=float, default=0.8); parser.add_argument("--global_ssim_weight", type=float, default=0.2)
    parser.add_argument("--global_contrast_weight", type=float, default=0.05); parser.add_argument("--region_l1_weight", type=float, default=1.0)
    parser.add_argument("--region_gradient_weight", type=float, default=0.2); parser.add_argument("--region_ssim_weight", type=float, default=0.2)
    parser.add_argument("--reconstruction_ssim_window", type=int, default=7); parser.add_argument("--reconstruction_min_valid_support", type=int, default=4)


def build_parser():
    parser = argparse.ArgumentParser("physical-mask-routed-source")
    add_model_arguments(parser); add_synthetic_data_arguments(parser); add_training_arguments(parser)
    parser.add_argument("--exp_dir", default=LOCAL_TEACHER_OUTPUT_DIR)
    parser.add_argument("--saved_model_dir", default=LOCAL_TEACHER_OUTPUT_DIR)
    parser.add_argument("--saved_data_dir", default="")
    parser.add_argument("--source_probe_hazy", default=LOCAL_SOURCE_PROBE_HAZY)
    parser.add_argument("--source_probe_tir", default=LOCAL_SOURCE_PROBE_TIR)
    parser.add_argument("--source_probe_output_dir", default=LOCAL_SOURCE_PROBE_OUTPUT_DIR)
    return parser


def validate_config(args):
    validate_model_config_values(vars(args))
    if args.density_gt_semantics != "density":
        raise ValueError("V2 physical mask routing requires density_gt_semantics='density'")
    for name in ("route_tau_start", "route_tau_end", "memory_attention_temperature", "density_smooth_l1_beta"):
        if getattr(args, name) <= 0: raise ValueError(f"{name} must be > 0")
    if args.route_temperature_anneal_steps < 0 or args.route_teacher_anneal_steps < 0: raise ValueError("route anneal steps must be non-negative")
    if args.train_size < 1 or min(args.batch_size, args.validation_batch_size) < 1 or args.num_workers < 0: raise ValueError("invalid Source batch configuration")
    if args.epochs < 1 or args.iters_per_epoch < 1 or min(args.start_lr, args.end_lr) <= 0: raise ValueError("invalid Source schedule")
    if any(getattr(args, key) < 0 for key in vars(args) if key.startswith("lambda_") or key.endswith("_weight")): raise ValueError("loss weights must be non-negative")
    if args.reconstruction_ssim_window <= 0 or args.reconstruction_ssim_window % 2 == 0: raise ValueError("reconstruction_ssim_window must be positive and odd")
    if args.reconstruction_min_valid_support < 1: raise ValueError("reconstruction_min_valid_support must be >= 1")
    if bool(args.source_probe_hazy) != bool(args.source_probe_tir):
        raise ValueError("source_probe_hazy and source_probe_tir must be provided together")
    if args.source_probe_hazy and not args.source_probe_output_dir:
        raise ValueError("source_probe_output_dir is required when a Source probe is configured")
    if args.tir_normalization == "fixed_range" and not args.tir_fixed_max > args.tir_fixed_min: raise ValueError("TIR fixed max must exceed min")
    if not 0 <= args.tir_percentile_low < args.tir_percentile_high <= 100: raise ValueError("invalid TIR percentiles")
    if args.density_map_normalization == "fixed_range" and not args.density_fixed_max > args.density_fixed_min: raise ValueError("density fixed max must exceed min")
    if args.density_map_normalization == "dataset_calibrated_range" and not args.density_calibrated_max > args.density_calibrated_min: raise ValueError("density calibrated max must exceed min")
    return args


def prepare_experiment_dirs(args):
    for value in (args.exp_dir, args.saved_model_dir, args.saved_data_dir, args.source_probe_output_dir):
        if value: Path(value).mkdir(parents=True, exist_ok=True)


def save_config(args):
    if args.exp_dir:
        Path(args.exp_dir).mkdir(parents=True, exist_ok=True)
        with (Path(args.exp_dir) / "config.json").open("w", encoding="utf-8") as handle:
            json.dump(persisted_config_from_args(args), handle, indent=2, sort_keys=True)
