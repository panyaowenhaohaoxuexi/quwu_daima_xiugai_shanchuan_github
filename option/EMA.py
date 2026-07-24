"""Pure EMA-adaptation configuration module; importing it has no side effects."""

import argparse
import json
from pathlib import Path

from ._formal_config import add_model_arguments, add_training_arguments, validate_common


def build_parser():
    parser = argparse.ArgumentParser("fog-routed-ema")
    add_model_arguments(parser)
    add_training_arguments(parser)
    parser.add_argument("--real_data_dir", default="")
    parser.add_argument("--source_checkpoint", default="")
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--real_batch_size", type=int, default=1)
    parser.add_argument("--source_anchor_batch_size", type=int, default=1)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--ema_sigma_j", type=float, default=0.1)
    parser.add_argument("--ema_sigma_m", type=float, default=0.1)
    parser.add_argument("--ema_sigma_r", type=float, default=0.1)
    parser.add_argument("--ema_stability_min_weight", type=float, default=0.05)
    parser.add_argument("--lambda_ema_j", type=float, default=1.0)
    parser.add_argument("--lambda_ema_m", type=float, default=1.0)
    parser.add_argument("--lambda_ema_r", type=float, default=1.0)
    parser.add_argument("--lambda_anchor", type=float, default=1.0)
    parser.add_argument("--allow_ema_training_override", action="store_true")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    return parser


def validate_config(args):
    validate_common(args)
    if not 0 <= args.ema_decay < 1:
        raise ValueError("ema_decay must be in [0,1)")
    if min(args.ema_sigma_j, args.ema_sigma_m, args.ema_sigma_r) <= 0:
        raise ValueError("EMA sigmas must be > 0")
    if not 0 <= args.ema_stability_min_weight <= 1:
        raise ValueError("ema_stability_min_weight must be in [0,1]")
    if args.real_batch_size < 1 or args.source_anchor_batch_size < 1:
        raise ValueError("EMA batch sizes must be >= 1")
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
        json.dump(vars(args), handle, indent=2, sort_keys=True)
