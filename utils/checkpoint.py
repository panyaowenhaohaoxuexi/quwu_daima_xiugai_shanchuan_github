"""Small strict checkpoints for Source, EMA and unified evaluation."""

import math
from collections.abc import Mapping
from numbers import Integral, Real

import torch

from .model_config_validation import (
    MODEL_CONFIG_KEYS,
    require_nonnegative_integer,
    require_positive_integer,
    validate_model_config_values,
)

CHECKPOINT_FORMAT_VERSION = 2
LEGACY_ROUTE_CONFIG_KEYS = frozenset({
    "boundary_width", "memory_exclusion_extra_margin", "counterfactual_chunk_size",
    "counterfactual_start_step", "route_loss_start_step", "route_loss_warmup_steps",
    "route_hard_start_step", "binary_loss_start_step", "binary_loss_warmup_steps",
    "omega_regions_per_image", "omega_min_area", "omega_max_area",
    "max_consecutive_empty_omega_steps", "q_temperature", "q_window_size",
    "q_min_valid_support", "lambda_binary", "lambda_global", "lambda_fuse",
    "lambda_comp", "lambda_boundary", "lambda_router",
})
LEGACY_ROUTER_STATE_MARKERS = ("router.raw_weight_in", "router.raw_weight_out", "router.bias_in", "router.bias_out")


def _legacy_v2_error(detail):
    return f"不兼容 V2 物理 mask 路由架构: {detail}"

def require_checkpoint_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a dictionary")


def require_checkpoint_format(checkpoint):
    require_checkpoint_dict(checkpoint)
    version = checkpoint.get("format_version")
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError("checkpoint format_version must be 2; "
                         f"received {version!r}. Legacy checkpoints are incompatible with the "
                         "structure-appearance Transformer decoder.")


def load_strict_v2_state_dict(model, state_dict, *, label):
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"{label} state_dict must be a mapping")
    if any(key in state_dict for key in LEGACY_ROUTER_STATE_MARKERS):
        raise RuntimeError(_legacy_v2_error("checkpoint contains MonotonicFogRouter parameters"))
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(_legacy_v2_error(f"{label} state_dict does not match FeatureGuidedRouter")) from exc


def require_complete_model_config(config):
    if not isinstance(config, Mapping):
        raise TypeError("checkpoint config must be a mapping")
    legacy = sorted(LEGACY_ROUTE_CONFIG_KEYS.intersection(config))
    router_name = str(config.get("router_class", config.get("router_type", "")))
    if legacy or router_name == "MonotonicFogRouter":
        detail = ("old-route config keys=" + ", ".join(legacy)) if legacy else "router_class=MonotonicFogRouter"
        raise ValueError(_legacy_v2_error(detail))
    missing = [key for key in MODEL_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError("checkpoint lacks model configuration: " + ", ".join(missing))
    validate_model_config_values(config)
    return config


def require_checkpoint_field(checkpoint, key, *, label=None):
    if key not in checkpoint:
        raise ValueError(f"checkpoint lacks {label or key}")
    return checkpoint[key]


def require_state_dict_field(checkpoint, key, *, label):
    """Validate a serialized state before any model construction occurs."""
    state_dict = require_checkpoint_field(checkpoint, key, label=f"{label} state")
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"{label} state_dict must be a mapping")
    if not state_dict:
        raise ValueError(f"{label} state_dict must not be empty")
    return state_dict


def require_integer_checkpoint_field(checkpoint, key, *, label=None):
    value = require_checkpoint_field(checkpoint, key, label=label or key)
    if isinstance(value, bool):
        raise TypeError(f"{label or key} must be an integer, received {value!r}")
    if isinstance(value, Real) and not isinstance(value, Integral):
        numeric = float(value)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise TypeError(f"{label or key} must be an integer, received {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(f"{label or key} must be an integer, received {value!r}") from exc


def require_optimizer_state_field(checkpoint, *, label):
    optimizer_state = require_checkpoint_field(checkpoint, "optimizer", label=f"{label} optimizer state")
    if not isinstance(optimizer_state, Mapping):
        raise TypeError(f"{label} optimizer state must be a mapping")
    return optimizer_state


def _preflight_checkpoint(checkpoint, *, stage, state_fields, integer_fields, optimizer_label=None):
    """Pure formal checkpoint validation shared by entries and strict loaders."""
    require_checkpoint_format(checkpoint)
    require_stage(checkpoint, stage)
    config = require_complete_model_config(require_checkpoint_field(checkpoint, "config", label="config"))
    states = {
        key: require_state_dict_field(checkpoint, key, label=label)
        for key, label in state_fields.items()
    }
    metadata = {
        key: require_integer_checkpoint_field(checkpoint, key)
        for key in integer_fields
    }
    optimizer_state = (
        require_optimizer_state_field(checkpoint, label=optimizer_label)
        if optimizer_label is not None else None
    )
    return {"config": config, "states": states, "metadata": metadata,
            "optimizer_state": optimizer_state}


def preflight_source_resume_checkpoint(checkpoint):
    return _preflight_checkpoint(
        checkpoint, stage="source", state_fields={"model": "Source model"},
        integer_fields=("epoch", "global_step"), optimizer_label="Source",
    )


def preflight_source_initialization_checkpoint(checkpoint):
    return _preflight_checkpoint(
        checkpoint, stage="source", state_fields={"model": "Source model"},
        integer_fields=("global_step",),
    )


def preflight_ema_resume_checkpoint(checkpoint):
    return _preflight_checkpoint(
        checkpoint, stage="ema", state_fields={"student": "EMA student", "teacher": "EMA teacher"},
        integer_fields=("epoch", "source_global_step", "ema_global_step"), optimizer_label="EMA",
    )


def preflight_eval_checkpoint(checkpoint, *, ema_model):
    require_checkpoint_format(checkpoint)
    stage = checkpoint.get("training_stage")
    if stage not in ("source", "ema"):
        raise ValueError("checkpoint training_stage must be 'source' or 'ema'")
    config = require_complete_model_config(require_checkpoint_field(checkpoint, "config", label="config"))
    state_key = "model" if stage == "source" else ema_model
    state_label = "Source model" if stage == "source" else f"EMA {ema_model}"
    return {"stage": stage, "config": config, "state_key": state_key,
            "state_label": state_label,
            "state_dict": require_state_dict_field(checkpoint, state_key, label=state_label)}


def build_model_from_config(config):
    require_complete_model_config(config)
    from model.Teacher import FogRoutedRGBTIRDehazer
    return FogRoutedRGBTIRDehazer(**{key: config[key] for key in MODEL_CONFIG_KEYS})


def build_source_checkpoint(model, optimizer, *, epoch, global_step, config, best_psnr=None):
    require_complete_model_config(config)
    checkpoint = {"format_version": CHECKPOINT_FORMAT_VERSION, "training_stage": "source", "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                  "epoch": int(epoch), "global_step": int(global_step), "config": dict(config)}
    if best_psnr is not None:
        checkpoint["best_psnr"] = float(best_psnr)
    return checkpoint


def build_ema_checkpoint(student, teacher, optimizer, *, epoch, source_global_step, ema_global_step, config, best_psnr=None):
    require_complete_model_config(config)
    checkpoint = {"format_version": CHECKPOINT_FORMAT_VERSION, "training_stage": "ema", "student": student.state_dict(), "teacher": teacher.state_dict(),
                  "optimizer": optimizer.state_dict(), "epoch": int(epoch),
                  "source_global_step": int(source_global_step), "ema_global_step": int(ema_global_step),
                  "config": dict(config)}
    if best_psnr is not None:
        checkpoint["best_psnr"] = float(best_psnr)
    return checkpoint


def require_stage(checkpoint, stage):
    require_checkpoint_dict(checkpoint)
    if checkpoint.get("training_stage") != stage:
        raise ValueError(f"checkpoint training_stage must be {stage!r}")


def load_source_checkpoint(checkpoint, model, optimizer=None):
    preflight = preflight_source_resume_checkpoint(checkpoint) if optimizer is not None else _preflight_checkpoint(
        checkpoint, stage="source", state_fields={"model": "Source model"},
        integer_fields=("epoch", "global_step"),
    )
    load_strict_v2_state_dict(model, preflight["states"]["model"], label="Source model")
    if optimizer is not None:
        try:
            optimizer.load_state_dict(preflight["optimizer_state"])
        except (KeyError, TypeError, RuntimeError, ValueError) as exc:
            raise RuntimeError("Source optimizer state is incompatible") from exc
    state = {"epoch": preflight["metadata"]["epoch"],
             "global_step": preflight["metadata"]["global_step"]}
    if "best_psnr" in checkpoint:
        state["best_psnr"] = float(checkpoint["best_psnr"])
    return state


def load_ema_checkpoint(checkpoint, student, teacher, optimizer=None):
    preflight = preflight_ema_resume_checkpoint(checkpoint) if optimizer is not None else _preflight_checkpoint(
        checkpoint, stage="ema", state_fields={"student": "EMA student", "teacher": "EMA teacher"},
        integer_fields=("epoch", "source_global_step", "ema_global_step"),
    )
    load_strict_v2_state_dict(student, preflight["states"]["student"], label="EMA student")
    load_strict_v2_state_dict(teacher, preflight["states"]["teacher"], label="EMA teacher")
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    if optimizer is not None:
        try:
            optimizer.load_state_dict(preflight["optimizer_state"])
        except (KeyError, TypeError, RuntimeError, ValueError) as exc:
            raise RuntimeError("EMA optimizer state is incompatible") from exc
    state = {"epoch": preflight["metadata"]["epoch"],
             "source_global_step": preflight["metadata"]["source_global_step"],
             "ema_global_step": preflight["metadata"]["ema_global_step"]}
    if "best_psnr" in checkpoint:
        state["best_psnr"] = float(checkpoint["best_psnr"])
    return state


def tir_normalization_config(config):
    return {"normalization": config["tir_normalization"], "fixed_min": config.get("tir_fixed_min"),
            "fixed_max": config.get("tir_fixed_max"), "percentile_low": config.get("tir_percentile_low"),
            "percentile_high": config.get("tir_percentile_high"),
            "percentile_scope": config.get("tir_percentile_scope"),
            "dataset_percentile_low_value": config.get("tir_dataset_percentile_low_value"),
            "dataset_percentile_high_value": config.get("tir_dataset_percentile_high_value"),
            "channel_tolerance_code_values": config.get("tir_channel_tolerance_code_values", 1),
            "channel_tolerance_float": config.get("tir_channel_tolerance_float", 1e-5)}
