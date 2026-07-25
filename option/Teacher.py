"""Pure source-training configuration module; importing it has no side effects."""

import argparse
import json
from pathlib import Path

from ._formal_config import (
    add_model_arguments, add_training_arguments, persisted_config_from_args, validate_common,
)


def build_parser():
    parser = argparse.ArgumentParser("fog-routed-source")
    add_model_arguments(parser)
    add_training_arguments(parser)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--allow_source_training_override", action="store_true")
    return parser


def validate_config(args):
    validate_common(args)
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
