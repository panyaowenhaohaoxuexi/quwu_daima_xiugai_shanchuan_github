"""Small strict checkpoints for Source, EMA and unified evaluation."""

from collections.abc import Mapping

import torch

from .model_config_validation import (
    MODEL_CONFIG_KEYS,
    require_nonnegative_integer,
    require_positive_integer,
    validate_model_config_values,
)

CHECKPOINT_FORMAT_VERSION = 2

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
    if not isinstance(state_dict, dict):
        raise TypeError(f"{label} state_dict must be a dictionary")
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"{label} state_dict is incompatible with the v2 structure-appearance Transformer architecture") from exc


def require_complete_model_config(config):
    if not isinstance(config, Mapping):
        raise TypeError("checkpoint config must be a mapping")
    missing = [key for key in MODEL_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError("checkpoint lacks model configuration: " + ", ".join(missing))
    validate_model_config_values(config)
    return config


def require_checkpoint_field(checkpoint, key, *, label=None):
    if key not in checkpoint:
        raise ValueError(f"checkpoint lacks {label or key}")
    return checkpoint[key]


def build_model_from_config(config):
    require_complete_model_config(config)
    from model.Teacher import FogRoutedRGBTIRDehazer
    return FogRoutedRGBTIRDehazer(**{key: config[key] for key in MODEL_CONFIG_KEYS})


def build_source_checkpoint(model, optimizer, *, epoch, global_step, config):
    require_complete_model_config(config)
    return {"format_version": CHECKPOINT_FORMAT_VERSION, "training_stage": "source", "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": int(epoch), "global_step": int(global_step), "config": dict(config)}


def build_ema_checkpoint(student, teacher, optimizer, *, epoch, source_global_step, ema_global_step, config):
    require_complete_model_config(config)
    return {"format_version": CHECKPOINT_FORMAT_VERSION, "training_stage": "ema", "student": student.state_dict(), "teacher": teacher.state_dict(),
            "optimizer": optimizer.state_dict(), "epoch": int(epoch),
            "source_global_step": int(source_global_step), "ema_global_step": int(ema_global_step),
            "config": dict(config)}


def require_stage(checkpoint, stage):
    require_checkpoint_dict(checkpoint)
    if checkpoint.get("training_stage") != stage:
        raise ValueError(f"checkpoint training_stage must be {stage!r}")


def load_source_checkpoint(checkpoint, model, optimizer=None):
    require_checkpoint_format(checkpoint)
    require_stage(checkpoint, "source")
    require_complete_model_config(require_checkpoint_field(checkpoint, "config", label="config"))
    load_strict_v2_state_dict(model, require_checkpoint_field(checkpoint, "model", label="Source model state"), label="Source model")
    if optimizer is not None:
        try:
            optimizer.load_state_dict(require_checkpoint_field(checkpoint, "optimizer", label="Source optimizer state"))
        except (TypeError, RuntimeError, ValueError) as exc:
            raise RuntimeError("Source optimizer state is incompatible") from exc
    return {"epoch": int(require_checkpoint_field(checkpoint, "epoch")),
            "global_step": int(require_checkpoint_field(checkpoint, "global_step"))}


def load_ema_checkpoint(checkpoint, student, teacher, optimizer=None):
    require_checkpoint_format(checkpoint)
    require_stage(checkpoint, "ema")
    require_complete_model_config(require_checkpoint_field(checkpoint, "config", label="config"))
    load_strict_v2_state_dict(student, require_checkpoint_field(checkpoint, "student", label="EMA student state"), label="EMA student")
    load_strict_v2_state_dict(teacher, require_checkpoint_field(checkpoint, "teacher", label="EMA teacher state"), label="EMA teacher")
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    if optimizer is not None:
        try:
            optimizer.load_state_dict(require_checkpoint_field(checkpoint, "optimizer", label="EMA optimizer state"))
        except (TypeError, RuntimeError, ValueError) as exc:
            raise RuntimeError("EMA optimizer state is incompatible") from exc
    return {"epoch": int(require_checkpoint_field(checkpoint, "epoch")),
            "source_global_step": int(require_checkpoint_field(checkpoint, "source_global_step")),
            "ema_global_step": int(require_checkpoint_field(checkpoint, "ema_global_step"))}


def tir_normalization_config(config):
    return {"normalization": config["tir_normalization"], "fixed_min": config.get("tir_fixed_min"),
            "fixed_max": config.get("tir_fixed_max"), "percentile_low": config.get("tir_percentile_low"),
            "percentile_high": config.get("tir_percentile_high"),
            "percentile_scope": config.get("tir_percentile_scope"),
            "dataset_percentile_low_value": config.get("tir_dataset_percentile_low_value"),
            "dataset_percentile_high_value": config.get("tir_dataset_percentile_high_value"),
            "channel_tolerance_code_values": config.get("tir_channel_tolerance_code_values", 1),
            "channel_tolerance_float": config.get("tir_channel_tolerance_float", 1e-5)}
