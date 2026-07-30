"""Configuration for two-stage FLIR-to-M3FD unsupervised domain adaptation."""

from option.EMA import (build_ema_checkpoint_config, build_parser as build_ema_parser,
                        persisted_config_from_args, validate_config as validate_ema_config)
from utils.model_config_validation import MODEL_CONFIG_KEYS


UDA_SEMANTIC_KEYS = (
    "stage", "style_probability", "style_beta_min", "style_beta_max",
    "style_min_gain", "style_max_gain", "style_max_abs_bias",
    "route_consistency_warmup_steps", "route_consistency_ramp_steps",
    "probe_hazy", "probe_tir", "probe_output_dir",
)


def build_parser():
    parser = build_ema_parser()
    parser.prog = "physical-mask-routed-uda"
    parser.add_argument("--stage", choices=("source_style", "ema"), default="source_style")
    parser.add_argument("--style_probability", type=float, default=0.5)
    parser.add_argument("--style_beta_min", type=float, default=0.0)
    parser.add_argument("--style_beta_max", type=float, default=0.6)
    parser.add_argument("--style_min_gain", type=float, default=0.75)
    parser.add_argument("--style_max_gain", type=float, default=1.35)
    parser.add_argument("--style_max_abs_bias", type=float, default=0.20)
    parser.add_argument("--route_consistency_warmup_steps", type=int, default=1000)
    parser.add_argument("--route_consistency_ramp_steps", type=int, default=1000)
    parser.add_argument("--probe_hazy", default="")
    parser.add_argument("--probe_tir", default="")
    parser.add_argument("--probe_output_dir", default="")
    return parser


def validate_config(args):
    if not 0.0 <= args.style_probability <= 1.0:
        raise ValueError("style_probability must be in [0, 1]")
    if not 0.0 <= args.style_beta_min <= args.style_beta_max <= 1.0:
        raise ValueError("style beta bounds must satisfy 0 <= min <= max <= 1")
    if not 0.0 < args.style_min_gain <= args.style_max_gain:
        raise ValueError("style gain bounds must satisfy 0 < min <= max")
    if args.style_max_abs_bias < 0.0:
        raise ValueError("style_max_abs_bias must be non-negative")
    if args.route_consistency_warmup_steps < 0 or args.route_consistency_ramp_steps < 0:
        raise ValueError("route consistency schedule steps must be non-negative")
    if bool(args.probe_hazy) != bool(args.probe_tir):
        raise ValueError("probe_hazy and probe_tir must be provided together")
    if all(key in vars(args) for key in MODEL_CONFIG_KEYS):
        validate_ema_config(args)
    return args


def build_uda_checkpoint_config(model_config, args):
    config = build_ema_checkpoint_config(model_config, args)
    current = persisted_config_from_args(args)
    return {**config, **{key: current[key] for key in UDA_SEMANTIC_KEYS}}
