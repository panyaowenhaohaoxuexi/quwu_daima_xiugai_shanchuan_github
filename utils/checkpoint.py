"""Small strict checkpoints for Source, EMA and unified evaluation."""

import torch

CHECKPOINT_FORMAT_VERSION = 2

MODEL_CONFIG_KEYS = (
    "base_channels", "router_hidden_channels", "deform_num_samples", "deform_max_offset",
    "num_structure_renderers", "memory_max_tokens", "memory_topk", "memory_query_chunk_size",
    "memory_attention_temperature", "memory_reliability_epsilon", "memory_reliable_ratio_threshold",
    "memory_confidence_threshold", "memory_exclusion_extra_margin", "boundary_width",
    "decoder_num_heads", "decoder_depth", "decoder_window_size", "decoder_window_chunk_size",
    "decoder_mlp_ratio", "decoder_attention_dropout", "decoder_projection_dropout", "decoder_ffn_dropout",
)


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


def build_model_from_config(config):
    missing = [key for key in MODEL_CONFIG_KEYS if key not in config]
    if missing:
        raise ValueError("checkpoint lacks model configuration: " + ", ".join(missing))
    from model.Teacher import FogRoutedRGBTIRDehazer
    return FogRoutedRGBTIRDehazer(**{key: config[key] for key in MODEL_CONFIG_KEYS})


def build_source_checkpoint(model, optimizer, *, epoch, global_step, config):
    return {"format_version": CHECKPOINT_FORMAT_VERSION, "training_stage": "source", "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": int(epoch), "global_step": int(global_step), "config": dict(config)}


def build_ema_checkpoint(student, teacher, optimizer, *, epoch, source_global_step, ema_global_step, config):
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
    load_strict_v2_state_dict(model, checkpoint.get("model"), label="Source model")
    if optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except (KeyError, RuntimeError, ValueError) as exc:
            raise RuntimeError("Source optimizer state is incompatible") from exc
    return {"epoch": int(checkpoint["epoch"]), "global_step": int(checkpoint["global_step"])}


def load_ema_checkpoint(checkpoint, student, teacher, optimizer=None):
    require_checkpoint_format(checkpoint)
    require_stage(checkpoint, "ema")
    load_strict_v2_state_dict(student, checkpoint.get("student"), label="EMA student")
    load_strict_v2_state_dict(teacher, checkpoint.get("teacher"), label="EMA teacher")
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    if optimizer is not None:
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
        except (KeyError, RuntimeError, ValueError) as exc:
            raise RuntimeError("EMA optimizer state is incompatible") from exc
    return {"epoch": int(checkpoint["epoch"]), "source_global_step": int(checkpoint["source_global_step"]),
            "ema_global_step": int(checkpoint["ema_global_step"])}


def tir_normalization_config(config):
    return {"normalization": config["tir_normalization"], "fixed_min": config.get("tir_fixed_min"),
            "fixed_max": config.get("tir_fixed_max"), "percentile_low": config.get("tir_percentile_low"),
            "percentile_high": config.get("tir_percentile_high"),
            "percentile_scope": config.get("tir_percentile_scope"),
            "dataset_percentile_low_value": config.get("tir_dataset_percentile_low_value"),
            "dataset_percentile_high_value": config.get("tir_dataset_percentile_high_value"),
            "channel_tolerance_code_values": config.get("tir_channel_tolerance_code_values", 1),
            "channel_tolerance_float": config.get("tir_channel_tolerance_float", 1e-5)}
