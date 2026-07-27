"""Checkpoint-resume configuration boundaries for Source training."""


LOSS_WEIGHT_NAMES = (
    "q_l1_weight", "q_gradient_weight", "q_ssim_weight",
    "rec_l1_weight", "rec_gradient_weight", "rec_ssim_weight",
    "boundary_l1_weight", "boundary_gradient_weight",
)
TRAINING_OBJECTIVE_KEYS = ("formal_training", *LOSS_WEIGHT_NAMES)

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
        raise ValueError(
            "source checkpoint training objective configuration is incomplete; missing: "
            + ", ".join(missing)
        )

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
