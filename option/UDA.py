"""Configuration for two-stage FLIR-to-M3FD unsupervised domain adaptation."""

from pathlib import Path

from option.EMA import (build_ema_checkpoint_config, build_parser as build_ema_parser,
                        persisted_config_from_args, resolve_ema_config,
                        validate_config as validate_ema_config)
from utils.model_config_validation import MODEL_CONFIG_KEYS


# ---------------------------------------------------------------------------
# Direct training configuration: edit this block, then run ``python UDA.py``.
# Stage A starts from the existing Source model.  After it is accepted, change
# only UDA_RUN_STAGE to "ema"; Stage B then loads Stage A's best checkpoint.
# ---------------------------------------------------------------------------
UDA_RUN_STAGE = "source_style"  # "source_style" first, then change to "ema"
UDA_DEVICE = "cuda"

UDA_FLIR_TRAIN_DIR = r"/root/autodl-tmp/1_FLIR/train"
UDA_FLIR_VALIDATION_DIR = r"/root/autodl-tmp/1_FLIR/test"
UDA_M3FD_ROOT_DIR = r"/root/autodl-tmp/2_M3FD_IVDehaze"
UDA_M3FD_TIR_DIR = r"/root/autodl-tmp/2_M3FD_IVDehaze/ir"
UDA_SOURCE_CHECKPOINT = r"/root/autodl-tmp/train_data_model/1_Teacher_train/source_best.pt"

UDA_STAGE_A_OUTPUT_DIR = r"/root/autodl-tmp/train_data_model/2_StageA_train"
UDA_STAGE_B_OUTPUT_DIR = r"/root/autodl-tmp/train_data_model/3_StageB_train"
UDA_PROBE_HAZY = r"/root/autodl-tmp/2_M3FD_IVDehaze/hazy/00896.png"
UDA_PROBE_TIR = r"/root/autodl-tmp/2_M3FD_IVDehaze/ir/00896.png"
# Optional Stage-A batch probe.  Files are paired by stem; ``vis-*`` hazy
# names also pair with ``ir-*`` TIR names.  Leave all three empty to disable.
UDA_SOURCE_PROBE_HAZY = r"/root/autodl-tmp/train_test/hazy"
UDA_SOURCE_PROBE_TIR = r"/root/autodl-tmp/train_test/ir"
UDA_SOURCE_PROBE_OUTPUT_DIR = r"/root/autodl-tmp/train_test/output"

UDA_EPOCHS = 10
UDA_ITERS_PER_EPOCH = 1000
UDA_NUM_WORKERS = 16
UDA_REAL_BATCH_SIZE = 8
UDA_SOURCE_ANCHOR_BATCH_SIZE = 8
UDA_VALIDATION_BATCH_SIZE = 8
UDA_START_LR = 1e-7
UDA_END_LR = 1e-8

UDA_STYLE_PROBABILITY = 0.5
UDA_STYLE_BETA_MIN = 0.0
UDA_STYLE_BETA_MAX = 0.6
UDA_STYLE_MIN_GAIN = 0.75
UDA_STYLE_MAX_GAIN = 1.35
UDA_STYLE_MAX_ABS_BIAS = 0.20
UDA_ROUTE_CONSISTENCY_WARMUP_STEPS = 1000
UDA_ROUTE_CONSISTENCY_RAMP_STEPS = 1000
UDA_EMA_DECAY = 0.95
UDA_EMA_SIGMA_J = 0.1
UDA_EMA_SIGMA_M = 0.1
UDA_EMA_SIGMA_R = 0.1
UDA_EMA_STABILITY_MIN_WEIGHT = 0.05
UDA_LAMBDA_EMA_J = 1.0
UDA_LAMBDA_EMA_M = 1.0
UDA_LAMBDA_EMA_R = 1.0
UDA_LAMBDA_ANCHOR = 1.0
UDA_CLIP_WEIGHT = 0.5


def _direct_defaults():
    """Return the no-CLI configuration for the currently selected UDA stage."""
    is_stage_a = UDA_RUN_STAGE == "source_style"
    output_dir = UDA_STAGE_A_OUTPUT_DIR if is_stage_a else UDA_STAGE_B_OUTPUT_DIR
    source_checkpoint = (UDA_SOURCE_CHECKPOINT if is_stage_a
                         else str(Path(UDA_STAGE_A_OUTPUT_DIR) / "source_style_best.pt"))
    return {
        "device": UDA_DEVICE, "stage": UDA_RUN_STAGE,
        "real_data_dir": UDA_M3FD_ROOT_DIR, "real_tir_dir": UDA_M3FD_TIR_DIR,
        "source_anchor_data_dir": UDA_FLIR_TRAIN_DIR, "validation_data_dir": UDA_FLIR_VALIDATION_DIR,
        "source_checkpoint": source_checkpoint, "resume_checkpoint": "",
        "real_batch_size": UDA_REAL_BATCH_SIZE, "source_anchor_batch_size": UDA_SOURCE_ANCHOR_BATCH_SIZE,
        "validation_batch_size": UDA_VALIDATION_BATCH_SIZE, "num_workers": UDA_NUM_WORKERS,
        "epochs": UDA_EPOCHS, "iters_per_epoch": UDA_ITERS_PER_EPOCH,
        "start_lr": UDA_START_LR, "end_lr": UDA_END_LR,
        "ema_decay": UDA_EMA_DECAY, "ema_sigma_j": UDA_EMA_SIGMA_J,
        "ema_sigma_m": UDA_EMA_SIGMA_M, "ema_sigma_r": UDA_EMA_SIGMA_R,
        "ema_stability_min_weight": UDA_EMA_STABILITY_MIN_WEIGHT,
        "lambda_ema_j": UDA_LAMBDA_EMA_J, "lambda_ema_m": UDA_LAMBDA_EMA_M,
        "lambda_ema_r": UDA_LAMBDA_EMA_R, "lambda_anchor": UDA_LAMBDA_ANCHOR,
        "w_loss_Clip": UDA_CLIP_WEIGHT,
        "exp_dir": output_dir, "saved_model_dir": output_dir,
        "saved_data_dir": str(Path(output_dir) / "diagnostics"),
        "style_probability": UDA_STYLE_PROBABILITY, "style_beta_min": UDA_STYLE_BETA_MIN,
        "style_beta_max": UDA_STYLE_BETA_MAX, "style_min_gain": UDA_STYLE_MIN_GAIN,
        "style_max_gain": UDA_STYLE_MAX_GAIN, "style_max_abs_bias": UDA_STYLE_MAX_ABS_BIAS,
        "route_consistency_warmup_steps": UDA_ROUTE_CONSISTENCY_WARMUP_STEPS,
        "route_consistency_ramp_steps": UDA_ROUTE_CONSISTENCY_RAMP_STEPS,
        "probe_hazy": UDA_PROBE_HAZY, "probe_tir": UDA_PROBE_TIR,
        "probe_output_dir": str(Path(output_dir) / "probe"),
        "source_probe_hazy": UDA_SOURCE_PROBE_HAZY,
        "source_probe_tir": UDA_SOURCE_PROBE_TIR,
        "source_probe_output_dir": UDA_SOURCE_PROBE_OUTPUT_DIR,
    }


UDA_SEMANTIC_KEYS = (
    "stage", "style_probability", "style_beta_min", "style_beta_max",
    "style_min_gain", "style_max_gain", "style_max_abs_bias",
    "route_consistency_warmup_steps", "route_consistency_ramp_steps",
    "probe_hazy", "probe_tir", "probe_output_dir",
    "source_probe_hazy", "source_probe_tir", "source_probe_output_dir",
)


def build_parser():
    parser = build_ema_parser()
    parser.prog = "physical-mask-routed-uda"
    parser.add_argument("--stage", choices=("source_style", "ema"), default=UDA_RUN_STAGE)
    parser.add_argument("--style_probability", type=float, default=UDA_STYLE_PROBABILITY)
    parser.add_argument("--style_beta_min", type=float, default=UDA_STYLE_BETA_MIN)
    parser.add_argument("--style_beta_max", type=float, default=UDA_STYLE_BETA_MAX)
    parser.add_argument("--style_min_gain", type=float, default=UDA_STYLE_MIN_GAIN)
    parser.add_argument("--style_max_gain", type=float, default=UDA_STYLE_MAX_GAIN)
    parser.add_argument("--style_max_abs_bias", type=float, default=UDA_STYLE_MAX_ABS_BIAS)
    parser.add_argument("--route_consistency_warmup_steps", type=int, default=UDA_ROUTE_CONSISTENCY_WARMUP_STEPS)
    parser.add_argument("--route_consistency_ramp_steps", type=int, default=UDA_ROUTE_CONSISTENCY_RAMP_STEPS)
    parser.add_argument("--probe_hazy", default=UDA_PROBE_HAZY)
    parser.add_argument("--probe_tir", default=UDA_PROBE_TIR)
    parser.add_argument("--probe_output_dir", default="")
    parser.add_argument("--source_probe_hazy", default=UDA_SOURCE_PROBE_HAZY)
    parser.add_argument("--source_probe_tir", default=UDA_SOURCE_PROBE_TIR)
    parser.add_argument("--source_probe_output_dir", default=UDA_SOURCE_PROBE_OUTPUT_DIR)
    parser.set_defaults(**_direct_defaults())
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
    if args.probe_hazy and not args.probe_output_dir:
        raise ValueError("probe_output_dir is required when a target probe is configured")
    if bool(args.source_probe_hazy) != bool(args.source_probe_tir):
        raise ValueError("source_probe_hazy and source_probe_tir must be provided together")
    if args.source_probe_hazy and not args.source_probe_output_dir:
        raise ValueError("source_probe_output_dir is required when a Source probe is configured")
    if all(key in vars(args) for key in MODEL_CONFIG_KEYS):
        validate_ema_config(args)
    return args


def resolve_uda_config(raw_args, checkpoint_config, *, resume=False):
    """Combine source architecture semantics with UDA runtime configuration.

    The model and Source-loss options must always be read from the compatible
    Source checkpoint.  UDA-only scheduling options stay user-configurable,
    except when resuming an EMA-stage UDA checkpoint.
    """
    raw = vars(raw_args)
    resolved = resolve_ema_config(raw_args, checkpoint_config, resume=resume)
    semantic_values = (
        {key: checkpoint_config[key] for key in UDA_SEMANTIC_KEYS}
        if resume else {key: raw[key] for key in UDA_SEMANTIC_KEYS}
    )
    return {**resolved, **semantic_values}


def build_uda_checkpoint_config(model_config, args):
    config = build_ema_checkpoint_config(model_config, args)
    current = persisted_config_from_args(args)
    return {**config, **{key: current[key] for key in UDA_SEMANTIC_KEYS}}
