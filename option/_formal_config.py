"""Shared pure parser helpers for formal fog-routed training entry points."""

import argparse


def tir_normalization_config_from_args(args):
    """Return the one TIR loader mapping shared by train/EMA/evaluation."""
    return {
        "normalization": args.tir_normalization,
        "fixed_min": args.tir_fixed_min,
        "fixed_max": args.tir_fixed_max,
        "percentile_low": args.tir_percentile_low,
        "percentile_high": args.tir_percentile_high,
        "percentile_scope": args.tir_percentile_scope,
        "channel_tolerance_code_values": args.tir_channel_tolerance_code_values,
        "channel_tolerance_float": args.tir_channel_tolerance_float,
    }


def add_model_arguments(parser: argparse.ArgumentParser):
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
    parser.add_argument("--tir_channel_tolerance_code_values", type=int, default=1)
    parser.add_argument("--tir_channel_tolerance_float", type=float, default=1e-5)
    parser.add_argument("--pair_alignment_policy", choices=("strict", "resize_tir_to_rgb"), default="strict")


def add_training_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--train_data_dir", default="")
    parser.add_argument("--train_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--counterfactual_chunk_size", type=int, default=4)
    parser.add_argument("--counterfactual_start_step", type=int, default=1000)
    parser.add_argument("--route_loss_start_step", type=int, default=1000)
    parser.add_argument("--route_loss_warmup_steps", type=int, default=1000)
    parser.add_argument("--binary_loss_start_step", type=int, default=5000)
    parser.add_argument("--binary_loss_warmup_steps", type=int, default=1000)
    parser.add_argument("--q_temperature", type=float, default=0.1)
    parser.add_argument("--q_window_size", type=int, default=7)
    parser.add_argument("--q_min_valid_support", type=int, default=4)
    parser.add_argument("--q_l1_weight", type=float, default=1.0)
    parser.add_argument("--q_gradient_weight", type=float, default=0.0)
    parser.add_argument("--q_ssim_weight", type=float, default=0.0)
    parser.add_argument("--omega_regions_per_image", type=int, default=6)
    parser.add_argument("--omega_min_area", type=int, default=16)
    parser.add_argument("--omega_max_area", type=int, default=256)
    parser.add_argument("--max_consecutive_empty_omega_steps", type=int, default=100)
    parser.add_argument("--density_smooth_l1_beta", type=float, default=0.1)
    parser.add_argument("--rec_l1_weight", type=float, default=1.0)
    parser.add_argument("--rec_gradient_weight", type=float, default=0.0)
    parser.add_argument("--rec_ssim_weight", type=float, default=0.0)
    parser.add_argument("--boundary_l1_weight", type=float, default=1.0)
    parser.add_argument("--boundary_gradient_weight", type=float, default=0.0)
    parser.add_argument("--reconstruction_ssim_window", type=int, default=7)
    parser.add_argument("--reconstruction_min_valid_support", type=int, default=4)
    for name in ("global", "fuse", "comp", "boundary", "router", "density", "route", "binary"):
        parser.add_argument(f"--lambda_{name}", type=float, default=1.0)
    parser.add_argument("--exp_dir", default="experiment")
    parser.add_argument("--saved_model_dir", default="")
    parser.add_argument("--saved_data_dir", default="")


def validate_common(args):
    positive = ("route_tau_start", "route_tau_end", "q_temperature", "density_smooth_l1_beta",
                "memory_attention_temperature")
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be > 0")
    if args.route_hard_start_step < 0 or args.counterfactual_start_step > args.route_loss_start_step:
        raise ValueError("invalid route/counterfactual schedule")
    if args.q_window_size <= 0 or args.q_window_size % 2 == 0:
        raise ValueError("q_window_size must be positive and odd")
    if args.q_min_valid_support < 1:
        raise ValueError("q_min_valid_support must be >= 1")
    if args.q_l1_weight < 0 or args.q_gradient_weight < 0 or args.q_ssim_weight < 0:
        raise ValueError("q loss weights must be non-negative")
    if args.reconstruction_ssim_window <= 0 or args.reconstruction_ssim_window % 2 == 0:
        raise ValueError("reconstruction_ssim_window must be positive and odd")
    if args.reconstruction_min_valid_support < 1:
        raise ValueError("reconstruction_min_valid_support must be >= 1")
    if not (4 <= args.omega_regions_per_image <= 8) or not (0 < args.omega_min_area <= args.omega_max_area):
        raise ValueError("invalid Omega configuration")
    if args.counterfactual_chunk_size < 1 or args.deform_num_samples < 1 or args.deform_max_offset < 0:
        raise ValueError("invalid counterfactual/deform configuration")
    if args.train_size < 1:
        raise ValueError("train_size must be >= 1")
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
        raise ValueError("boundary_width and max_consecutive_empty_omega_steps must be >= 1")
    if any(getattr(args, name) < 0 for name in vars(args) if name.startswith("lambda_")):
        raise ValueError("lambda coefficients must be non-negative")
    if args.tir_normalization == "fixed_range" and not (args.tir_fixed_max > args.tir_fixed_min):
        raise ValueError("TIR fixed max must exceed min")
    if not (0 <= args.tir_percentile_low < args.tir_percentile_high <= 100):
        raise ValueError("invalid TIR percentiles")
    if args.density_map_normalization == "fixed_range" and not (args.density_fixed_max > args.density_fixed_min):
        raise ValueError("density fixed max must exceed min")
    if args.density_map_normalization == "dataset_calibrated_range" and not (args.density_calibrated_max > args.density_calibrated_min):
        raise ValueError("density calibrated max must exceed min")
