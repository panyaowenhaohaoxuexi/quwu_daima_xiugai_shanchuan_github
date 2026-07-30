"""Configuration for V2 EMA adaptation with Source physical-mask anchors."""

import argparse
import json
from pathlib import Path

from utils.model_config_validation import MODEL_CONFIG_KEYS, validate_model_config_values


LOCAL_SOURCE_TRAIN_DIR = r"/root/autodl-tmp/1_FLIR/train"
LOCAL_VALIDATION_DATA_DIR = r"/root/autodl-tmp/1_FLIR/test"
LOCAL_SOURCE_CHECKPOINT = r"/root/autodl-tmp/train_data_model/1_Teacher_train/source_last.pt"
LOCAL_REAL_ROOT_DIR = r"/root/autodl-tmp/2_M3FD"
LOCAL_REAL_TIR_DIR = r"/root/autodl-tmp/2_M3FD/ir"
LOCAL_EMA_OUTPUT_DIR = r"/root/autodl-tmp/train_data_model/2_Student_train"

INHERITED_SOURCE_KEYS = (*MODEL_CONFIG_KEYS, "model_init_seed", "route_tau_start", "route_tau_end",
                         "route_temperature_anneal_steps", "route_teacher_anneal_steps", "train_size",
                         "density_gt_semantics", "density_map_normalization", "density_fixed_min", "density_fixed_max",
                         "density_calibrated_min", "density_calibrated_max", "tir_normalization", "tir_fixed_min",
                         "tir_fixed_max", "tir_percentile_low", "tir_percentile_high", "tir_percentile_scope",
                         "tir_dataset_percentile_low_value", "tir_dataset_percentile_high_value",
                         "tir_channel_tolerance_code_values", "tir_channel_tolerance_float", "pair_alignment_policy",
                         "density_smooth_l1_beta", "lambda_density", "lambda_route", "lambda_global",
                         "lambda_fuse", "lambda_comp", "lambda_boundary", "global_l1_weight",
                         "global_ssim_weight", "global_contrast_weight", "region_l1_weight",
                         "region_gradient_weight", "region_ssim_weight", "reconstruction_ssim_window",
                         "reconstruction_min_valid_support")
EMA_RUNTIME_KEYS = ("device", "real_data_dir", "real_tir_dir", "source_anchor_data_dir", "validation_data_dir",
                    "source_checkpoint", "resume_checkpoint", "real_batch_size", "source_anchor_batch_size",
                    "validation_batch_size", "num_workers", "epochs", "iters_per_epoch", "start_lr", "end_lr",
                    "no_lr_sche", "exp_dir", "saved_model_dir", "saved_data_dir")
EMA_SEMANTIC_KEYS = ("ema_decay", "ema_sigma_j", "ema_sigma_m", "ema_sigma_r", "ema_stability_min_weight",
                     "lambda_ema_j", "lambda_ema_m", "lambda_ema_r", "lambda_anchor", "w_loss_Clip")
EMA_CHECKPOINT_CONFIG_KEYS = (*INHERITED_SOURCE_KEYS, *EMA_SEMANTIC_KEYS, "real_data_dir", "real_tir_dir",
                              "source_anchor_data_dir", "validation_data_dir", "real_batch_size",
                              "source_anchor_batch_size", "validation_batch_size", "num_workers", "epochs",
                              "iters_per_epoch", "start_lr", "end_lr", "no_lr_sche", "exp_dir",
                              "saved_model_dir", "saved_data_dir")


def persisted_config_from_args(args): return {key: value for key, value in vars(args).items() if not key.startswith("_")}

def tir_normalization_config_from_args(args):
    return {"normalization": args.tir_normalization, "fixed_min": args.tir_fixed_min, "fixed_max": args.tir_fixed_max,
            "percentile_low": args.tir_percentile_low, "percentile_high": args.tir_percentile_high,
            "percentile_scope": args.tir_percentile_scope, "dataset_percentile_low_value": args.tir_dataset_percentile_low_value,
            "dataset_percentile_high_value": args.tir_dataset_percentile_high_value,
            "channel_tolerance_code_values": args.tir_channel_tolerance_code_values,
            "channel_tolerance_float": args.tir_channel_tolerance_float}

def real_modal_dirs_from_args(args): return str(Path(args.real_data_dir) / "hazy"), args.real_tir_dir

def build_parser():
    parser = argparse.ArgumentParser("physical-mask-routed-ema")
    parser.add_argument("--device", default="cuda"); parser.add_argument("--real_data_dir", default=LOCAL_REAL_ROOT_DIR)
    parser.add_argument("--real_tir_dir", default=LOCAL_REAL_TIR_DIR); parser.add_argument("--source_anchor_data_dir", default=LOCAL_SOURCE_TRAIN_DIR)
    parser.add_argument("--validation_data_dir", default=LOCAL_VALIDATION_DATA_DIR); parser.add_argument("--source_checkpoint", default=LOCAL_SOURCE_CHECKPOINT)
    parser.add_argument("--resume_checkpoint", default=""); parser.add_argument("--real_batch_size", type=int, default=1)
    parser.add_argument("--source_anchor_batch_size", type=int, default=1); parser.add_argument("--validation_batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0); parser.add_argument("--epochs", type=int, default=20); parser.add_argument("--iters_per_epoch", type=int, default=1000)
    parser.add_argument("--start_lr", "--learning_rate", dest="start_lr", type=float, default=1e-7); parser.add_argument("--end_lr", type=float, default=1e-8)
    parser.add_argument("--no_lr_sche", action="store_true"); parser.add_argument("--ema_decay", type=float, default=0.95)
    parser.add_argument("--ema_sigma_j", type=float, default=0.1); parser.add_argument("--ema_sigma_m", type=float, default=0.1); parser.add_argument("--ema_sigma_r", type=float, default=0.1)
    parser.add_argument("--ema_stability_min_weight", type=float, default=0.05); parser.add_argument("--lambda_ema_j", type=float, default=1.0)
    parser.add_argument("--lambda_ema_m", type=float, default=1.0); parser.add_argument("--lambda_ema_r", type=float, default=1.0); parser.add_argument("--lambda_anchor", type=float, default=1.0)
    parser.add_argument("--w_loss_Clip", type=float, default=0.5); parser.add_argument("--exp_dir", default=LOCAL_EMA_OUTPUT_DIR)
    parser.add_argument("--saved_model_dir", default=LOCAL_EMA_OUTPUT_DIR); parser.add_argument("--saved_data_dir", default="")
    return parser

def _inherited_source_config(checkpoint_config):
    if not isinstance(checkpoint_config, dict): raise TypeError("checkpoint config must be a dictionary")
    missing = [key for key in INHERITED_SOURCE_KEYS if key not in checkpoint_config]
    if missing: raise ValueError("incompatible V2 physical mask routing architecture: checkpoint lacks required Source configuration: " + ", ".join(missing))
    return {key: checkpoint_config[key] for key in INHERITED_SOURCE_KEYS}

def resolve_ema_config(raw_args, checkpoint_config, *, resume=False):
    raw = vars(raw_args) if isinstance(raw_args, argparse.Namespace) else dict(raw_args)
    inherited = _inherited_source_config(checkpoint_config)
    semantics = ({key: checkpoint_config[key] for key in EMA_SEMANTIC_KEYS} if resume else {key: raw[key] for key in EMA_SEMANTIC_KEYS})
    return {**inherited, **semantics, **{key: raw[key] for key in EMA_RUNTIME_KEYS}}

def validate_config(args):
    validate_model_config_values(vars(args))
    if args.density_gt_semantics != "density": raise ValueError("V2 physical mask routing requires density_gt_semantics='density'")
    if args.route_tau_start <= 0 or args.route_tau_end <= 0 or args.density_smooth_l1_beta <= 0: raise ValueError("V2 route temperatures and density beta must be > 0")
    if args.route_temperature_anneal_steps < 0 or args.route_teacher_anneal_steps < 0: raise ValueError("route anneal steps must be non-negative")
    if not all((args.real_data_dir, args.real_tir_dir, args.source_anchor_data_dir, args.validation_data_dir)): raise ValueError("EMA data directories must be provided")
    if not 0 <= args.ema_decay < 1 or not 0 <= args.ema_stability_min_weight <= 1: raise ValueError("invalid EMA decay or stability weight")
    if min(args.ema_sigma_j, args.ema_sigma_m, args.ema_sigma_r) <= 0: raise ValueError("EMA sigmas must be > 0")
    if min(args.real_batch_size, args.source_anchor_batch_size, args.validation_batch_size, args.epochs, args.iters_per_epoch) < 1 or args.num_workers < 0: raise ValueError("invalid EMA batch configuration")
    if min(args.start_lr, args.end_lr) <= 0 or any(getattr(args, key) < 0 for key in ("lambda_ema_j", "lambda_ema_m", "lambda_ema_r", "lambda_anchor", "w_loss_Clip", "lambda_density", "lambda_route")): raise ValueError("EMA weights must be non-negative")
    return args

def build_ema_checkpoint_config(model_config, args):
    inherited = _inherited_source_config(model_config); current = persisted_config_from_args(args)
    return {**inherited, **{key: current[key] for key in EMA_CHECKPOINT_CONFIG_KEYS}}

def prepare_experiment_dirs(args):
    for value in (args.exp_dir, args.saved_model_dir, args.saved_data_dir):
        if value: Path(value).mkdir(parents=True, exist_ok=True)

def save_config(args):
    if args.exp_dir:
        Path(args.exp_dir).mkdir(parents=True, exist_ok=True)
        with (Path(args.exp_dir) / "config.json").open("w", encoding="utf-8") as handle: json.dump(persisted_config_from_args(args), handle, indent=2, sort_keys=True)
