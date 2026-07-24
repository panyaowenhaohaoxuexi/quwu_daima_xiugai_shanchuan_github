"""Strict formal checkpoint metadata and reproducible RNG helpers."""

import random

import numpy as np
import torch


_MODEL_CONSTRUCTION_KEYS = (
    "base_channels", "router_hidden_channels", "deform_num_samples", "deform_max_offset",
    "num_structure_renderers", "memory_max_tokens", "memory_topk",
    "memory_query_chunk_size",
    "memory_attention_temperature", "memory_reliability_epsilon",
    "memory_reliable_ratio_threshold", "memory_confidence_threshold",
    "memory_exclusion_extra_margin", "boundary_width",
)


def capture_rng_state(omega_generator=None, geometry_generator=None, dataloader_generators=None):
    return {
        "python_random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "omega_generator": omega_generator.get_state() if omega_generator is not None else None,
        "geometry_generator": geometry_generator.get_state() if geometry_generator is not None else None,
        "dataloader_generators": {
            key: value.get_state() for key, value in (dataloader_generators or {}).items()
        },
    }


def restore_rng_state(state, omega_generator=None, geometry_generator=None, dataloader_generators=None):
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda_all") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda_all"])
    if omega_generator is not None and state.get("omega_generator") is not None:
        omega_generator.set_state(state["omega_generator"])
    if geometry_generator is not None and state.get("geometry_generator") is not None:
        geometry_generator.set_state(state["geometry_generator"])
    for key, generator in (dataloader_generators or {}).items():
        if key in state.get("dataloader_generators", {}):
            generator.set_state(state["dataloader_generators"][key])


def validate_checkpoint_metadata(checkpoint, expected_stage, density_semantics):
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported or legacy checkpoint format")
    if checkpoint.get("training_stage") != expected_stage:
        raise ValueError(f"checkpoint stage must be {expected_stage}")
    if checkpoint.get("model_class") != "FogRoutedRGBTIRDehazer":
        raise ValueError("checkpoint model_class mismatch")
    if density_semantics not in ("transmission", "density"):
        raise ValueError("checkpoint lacks valid density semantics")
    if checkpoint.get("density_gt_semantics") != density_semantics:
        raise ValueError("checkpoint density semantics mismatch")


def build_formal_model_from_config(config):
    """Construct the formal architecture solely from persisted semantics."""
    if not isinstance(config, dict):
        raise TypeError("checkpoint config must be a dictionary")
    missing = [key for key in _MODEL_CONSTRUCTION_KEYS if key not in config]
    if missing:
        raise ValueError(f"checkpoint lacks model semantic configuration: {', '.join(missing)}")
    # Local import keeps checkpoint/RNG helpers pure until construction is
    # explicitly requested by a training or evaluation entry point.
    from model import FogRoutedRGBTIRDehazer

    return FogRoutedRGBTIRDehazer(**{key: config[key] for key in _MODEL_CONSTRUCTION_KEYS})


def validate_model_semantics_config(stored_config, current_config):
    """Reject resume-time architecture semantics drift, even when shapes match."""
    differences = {
        key: (stored_config.get(key), current_config.get(key))
        for key in _MODEL_CONSTRUCTION_KEYS
        if stored_config.get(key) != current_config.get(key)
    }
    if differences:
        rendered = ", ".join(f"{key}: {old!r}->{new!r}" for key, (old, new) in differences.items())
        raise ValueError(f"semantic configuration mismatch: {rendered}")


def tir_normalization_config_from_checkpoint(config):
    """Translate persisted CLI semantics into the shared TIR loader mapping."""
    if not isinstance(config, dict) or "tir_normalization" not in config:
        raise ValueError("checkpoint lacks TIR normalization semantics")
    return {
        "normalization": config["tir_normalization"],
        "fixed_min": config.get("tir_fixed_min"),
        "fixed_max": config.get("tir_fixed_max"),
        "percentile_low": config.get("tir_percentile_low"),
        "percentile_high": config.get("tir_percentile_high"),
        "percentile_scope": config.get("tir_percentile_scope"),
        "dataset_percentile_low_value": config.get("tir_dataset_percentile_low_value"),
        "dataset_percentile_high_value": config.get("tir_dataset_percentile_high_value"),
        "channel_tolerance_code_values": config.get("tir_channel_tolerance_code_values", 1),
        "channel_tolerance_float": config.get("tir_channel_tolerance_float", 1e-5),
    }


def _metadata(stage, config, density_semantics, rng_state):
    return {
        "format_version": 1,
        "training_stage": stage,
        "model_class": "FogRoutedRGBTIRDehazer",
        "config": config,
        "density_gt_semantics": density_semantics,
        "rng_state": rng_state,
    }


def build_source_checkpoint(model, optimizer, scheduler, global_step, epoch, config,
                            density_semantics, rng_state, *, sampler_states=None,
                            empty_omega_streaks=None, manifest_fingerprints=None):
    checkpoint = _metadata("source", config, density_semantics, rng_state)
    checkpoint.update({
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "global_step": int(global_step),
        "epoch": int(epoch),
        "sampler_states": dict(sampler_states or {}),
        "empty_omega_streaks": dict(empty_omega_streaks or {}),
        "manifest_fingerprints": dict(manifest_fingerprints or {}),
    })
    return checkpoint


def build_ema_checkpoint(student, teacher, optimizer, scheduler, source_global_step,
                         ema_global_step, epoch, config, density_semantics, rng_state, *,
                         sampler_states=None, empty_omega_streaks=None, manifest_fingerprints=None):
    checkpoint = _metadata("ema", config, density_semantics, rng_state)
    checkpoint.update({
        "student": student,
        "teacher": teacher,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "source_global_step": int(source_global_step),
        "ema_global_step": int(ema_global_step),
        "epoch": int(epoch),
        "sampler_states": dict(sampler_states or {}),
        "empty_omega_streaks": dict(empty_omega_streaks or {}),
        "manifest_fingerprints": dict(manifest_fingerprints or {}),
    })
    return checkpoint


def restore_source_training_state(checkpoint, model, optimizer, sampler, density_semantics,
                                  scheduler=None, manifest_fingerprint=None, omega_generator=None,
                                  dataloader_generators=None):
    """Strictly restore a source checkpoint before creating the next iterator."""
    validate_checkpoint_metadata(checkpoint, "source", density_semantics)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    source_state = checkpoint.get("sampler_states", {}).get("source")
    if source_state is None:
        raise ValueError("source checkpoint lacks source sampler state")
    sampler.load_state_dict(source_state)
    if manifest_fingerprint is not None and checkpoint.get("manifest_fingerprints", {}).get("source") != manifest_fingerprint:
        raise ValueError("source dataset manifest fingerprint mismatch")
    restore_rng_state(
        checkpoint["rng_state"], omega_generator=omega_generator,
        dataloader_generators=dataloader_generators,
    )
    return {
        "global_step": int(checkpoint["global_step"]),
        "epoch": int(checkpoint["epoch"]),
        "empty_omega_streak": int(checkpoint.get("empty_omega_streaks", {}).get("source", 0)),
    }


def restore_ema_training_state(checkpoint, student, teacher, optimizer, source_sampler,
                               real_sampler, density_semantics, scheduler=None,
                               manifest_fingerprints=None, omega_generator=None,
                               geometry_generator=None, dataloader_generators=None):
    """Strictly restore the paired EMA state before either iterator is created."""
    validate_checkpoint_metadata(checkpoint, "ema", density_semantics)
    student.load_state_dict(checkpoint["student"], strict=True)
    teacher.load_state_dict(checkpoint["teacher"], strict=True)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and checkpoint.get("scheduler") is not None:
        scheduler.load_state_dict(checkpoint["scheduler"])
    sampler_states = checkpoint.get("sampler_states", {})
    if not {"source", "real"}.issubset(sampler_states):
        raise ValueError("EMA checkpoint lacks source/real sampler state")
    source_sampler.load_state_dict(sampler_states["source"])
    real_sampler.load_state_dict(sampler_states["real"])
    if manifest_fingerprints is not None:
        stored = checkpoint.get("manifest_fingerprints", {})
        if stored != manifest_fingerprints:
            raise ValueError("EMA dataset manifest fingerprint mismatch")
    restore_rng_state(
        checkpoint["rng_state"], omega_generator=omega_generator,
        geometry_generator=geometry_generator, dataloader_generators=dataloader_generators,
    )
    streaks = checkpoint.get("empty_omega_streaks", {})
    return {
        "source_global_step": int(checkpoint["source_global_step"]),
        "ema_global_step": int(checkpoint["ema_global_step"]),
        "epoch": int(checkpoint["epoch"]),
        "source_empty_omega_streak": int(streaks.get("source", 0)),
        "anchor_empty_omega_streak": int(streaks.get("anchor", 0)),
    }
