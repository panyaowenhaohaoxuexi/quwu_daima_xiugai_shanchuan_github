"""Pure source-training configuration module; importing it has no side effects."""

import argparse
import json
from pathlib import Path

from ._formal_config import add_model_arguments, add_training_arguments, persisted_config_from_args, validate_common


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


def add_source_loss_arguments(parser):
    """Register the authoritative source and EMA-anchor objective parameters."""
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


def validate_source_loss_arguments(args):
    """Validate Source/anchor loss parameters without reading global args."""
    for name in ("q_temperature", "density_smooth_l1_beta"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be > 0")
    if args.q_window_size <= 0 or args.q_window_size % 2 == 0:
        raise ValueError("q_window_size must be positive and odd")
    if args.q_min_valid_support < 1:
        raise ValueError("q_min_valid_support must be >= 1")
    negative = [name for name in LOSS_WEIGHT_NAMES if getattr(args, name) < 0]
    if negative:
        raise ValueError("loss weights must be non-negative: " + ", ".join(negative))
    if args.formal_training:
        explicit = set(getattr(args, "_explicit_training_objective_keys", ()))
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
    add_training_arguments(parser)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--resume_checkpoint", default="")
    parser.add_argument("--allow_source_training_override", action="store_true")
    add_source_loss_arguments(parser)
    return parser


def validate_config(args):
    validate_common(args)
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
