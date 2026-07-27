"""Checkpoint-resume configuration boundaries for formal training stages."""

import warnings


LOSS_WEIGHT_NAMES = (
    "q_l1_weight", "q_gradient_weight", "q_ssim_weight",
    "rec_l1_weight", "rec_gradient_weight", "rec_ssim_weight",
    "boundary_l1_weight", "boundary_gradient_weight",
)
TRAINING_OBJECTIVE_KEYS = ("formal_training", *LOSS_WEIGHT_NAMES)
_LEGACY_SOURCE_OBJECTIVE_DEFAULTS = {
    "formal_training": False,
    "q_l1_weight": 1.0, "q_gradient_weight": 0.0, "q_ssim_weight": 0.0,
    "rec_l1_weight": 1.0, "rec_gradient_weight": 0.0, "rec_ssim_weight": 0.0,
    "boundary_l1_weight": 1.0, "boundary_gradient_weight": 0.0,
}

_SEMANTIC_KEYS = {
    "base_channels", "router_hidden_channels", "deform_num_samples", "deform_max_offset",
    "num_structure_renderers", "memory_max_tokens", "memory_topk", "memory_query_chunk_size",
    "memory_attention_temperature", "memory_reliability_epsilon",
    "memory_reliable_ratio_threshold", "memory_confidence_threshold",
    "memory_exclusion_extra_margin", "boundary_width", "route_tau_end",
    "tir_normalization", "tir_fixed_min", "tir_fixed_max", "tir_percentile_low",
    "tir_percentile_high", "tir_percentile_scope", "tir_dataset_percentile_low_value",
    "tir_dataset_percentile_high_value", "tir_channel_tolerance_code_values",
    "tir_channel_tolerance_float", "density_gt_semantics", "density_map_normalization",
    "density_fixed_min", "density_fixed_max", "density_calibrated_min",
    "density_calibrated_max", "pair_alignment_policy", "train_size",
}

# These control where or how a resumed process is run, rather than the saved
# model/preprocessing semantics or EMA objective.  They intentionally remain
# under the resumed invocation's control: in particular a new data-root is
# necessary when a portable checkpoint was created with a relative/empty path,
# and ``epochs`` is the requested stop boundary for the current invocation.
_EMA_RUNTIME_KEYS = {
    "device", "num_workers", "exp_dir", "saved_model_dir", "saved_data_dir",
    "log_dir", "output_dir", "epochs", "source_anchor_data_dir", "real_data_dir",
    "source_checkpoint", "resume_checkpoint", "allow_ema_training_override",
    "log_frequency", "save_frequency", "save_visualizations",
}


def apply_checkpoint_semantics(current, checkpoint_config):
    """Return CLI/runtime values with all persisted model/preprocess semantics locked."""
    resolved = dict(current)
    for key in _SEMANTIC_KEYS:
        if key in checkpoint_config and key in resolved:
            resolved[key] = checkpoint_config[key]
    return resolved


def validate_source_resume_semantics(checkpoint_config, current):
    """Reject source-resume changes to architecture *and* preprocessing meaning."""
    differences = {
        key: (checkpoint_config.get(key), current.get(key))
        for key in _SEMANTIC_KEYS
        if checkpoint_config.get(key) != current.get(key)
    }
    if differences:
        rendered = ", ".join(
            f"{key}: {stored!r}->{requested!r}"
            for key, (stored, requested) in sorted(differences.items())
        )
        raise ValueError(f"source resume semantic configuration mismatch: {rendered}")


def apply_source_resume_config(current, checkpoint_config, *, allow_training_override,
                               explicit_objective_keys):
    """Lock source-training objectives unless an explicit override is allowed."""
    missing = [key for key in TRAINING_OBJECTIVE_KEYS if key not in checkpoint_config]
    if missing:
        warnings.warn(
            "legacy source checkpoint lacks training objective configuration; "
            "using historical L1 defaults for: " + ", ".join(missing),
            RuntimeWarning,
            stacklevel=2,
        )
        checkpoint_config = dict(checkpoint_config)
        checkpoint_config.update({key: _LEGACY_SOURCE_OBJECTIVE_DEFAULTS[key] for key in missing})
        # A partial formal profile cannot be validated without inventing a
        # composite objective.  Preserve every recorded weight, but disable
        # preset application so missing fields retain historical L1 values.
        if any(key in LOSS_WEIGHT_NAMES for key in missing):
            checkpoint_config["formal_training"] = False

    resolved = dict(current)
    explicit = set(explicit_objective_keys).intersection(TRAINING_OBJECTIVE_KEYS)
    diff = {}
    for key in TRAINING_OBJECTIVE_KEYS:
        checkpoint_value = checkpoint_config[key]
        current_value = current[key]
        if allow_training_override and key in explicit:
            resolved[key] = current_value
            if checkpoint_value != current_value:
                diff[key] = (checkpoint_value, current_value)
        else:
            resolved[key] = checkpoint_value

    # Formal validation treats unmarked weights as preset candidates. Mark all
    # resolved checkpoint weights so a source resume preserves saved custom
    # objectives rather than replacing them with the formal preset.
    preserved = set(explicit)
    preserved.update(LOSS_WEIGHT_NAMES)
    resolved["_explicit_training_objective_keys"] = sorted(preserved)
    return resolved, diff


def apply_ema_resume_config(current, checkpoint_config, *, allow_training_override):
    """Lock checkpoint semantics; optionally retain only EMA-training overrides.

    The function operates on dictionaries so it can be unit-tested before the
    training entry point mutates an argparse namespace.
    """
    resolved = apply_checkpoint_semantics(current, checkpoint_config)
    if not allow_training_override:
        for key, value in checkpoint_config.items():
            if key in resolved and key not in _EMA_RUNTIME_KEYS:
                resolved[key] = value
        return resolved, {}
    differences = {
        key: (checkpoint_config[key], current[key])
        for key in checkpoint_config
        if (
            key in current
            and key not in _SEMANTIC_KEYS
            and key not in _EMA_RUNTIME_KEYS
            and checkpoint_config[key] != current[key]
        )
    }
    return resolved, differences
