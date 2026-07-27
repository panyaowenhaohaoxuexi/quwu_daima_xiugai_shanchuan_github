"""Side-effect-free validation for formal model checkpoint semantics."""

import math
from numbers import Integral, Real


MODEL_CONFIG_KEYS = (
    "base_channels", "router_hidden_channels", "deform_num_samples", "deform_max_offset",
    "num_structure_renderers", "memory_max_tokens", "memory_topk", "memory_query_chunk_size",
    "memory_attention_temperature", "memory_reliability_epsilon", "memory_reliable_ratio_threshold",
    "memory_confidence_threshold", "memory_exclusion_extra_margin", "boundary_width",
    "decoder_num_heads", "decoder_depth", "decoder_window_size", "decoder_window_chunk_size",
    "decoder_mlp_ratio", "decoder_attention_dropout", "decoder_projection_dropout", "decoder_ffn_dropout",
)


def require_positive_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer, received {value!r}")
    return int(value)


def require_nonnegative_integer(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer, received {value!r}")
    return int(value)


def require_finite_float(name, value, *, minimum=None, maximum=None, upper_inclusive=True):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number, received {value!r}")
    converted = float(value)
    if minimum is not None and converted < minimum:
        raise ValueError(f"{name} must be >= {minimum}, received {value!r}")
    if maximum is not None and (converted > maximum or (not upper_inclusive and converted >= maximum)):
        operator = "<" if not upper_inclusive else "<="
        raise ValueError(f"{name} must be {operator} {maximum}, received {value!r}")
    return converted


def validate_model_config_values(config):
    """Validate only architecture/state-dict semantics without constructing a model."""
    for name in (
        "base_channels", "router_hidden_channels", "deform_num_samples", "num_structure_renderers",
        "memory_max_tokens", "memory_topk", "memory_query_chunk_size", "boundary_width",
        "decoder_num_heads", "decoder_depth", "decoder_window_size", "decoder_window_chunk_size",
    ):
        require_positive_integer(name, config[name])
    require_nonnegative_integer("memory_exclusion_extra_margin", config["memory_exclusion_extra_margin"])
    for name in ("deform_max_offset",):
        require_finite_float(name, config[name], minimum=0.0)
    for name in ("memory_attention_temperature", "memory_reliability_epsilon", "decoder_mlp_ratio"):
        value = require_finite_float(name, config[name])
        if value <= 0.0:
            raise ValueError(f"{name} must be > 0, received {config[name]!r}")
    for name in ("memory_reliable_ratio_threshold", "memory_confidence_threshold"):
        require_finite_float(name, config[name], minimum=0.0, maximum=1.0)
    for name in ("decoder_attention_dropout", "decoder_projection_dropout", "decoder_ffn_dropout"):
        require_finite_float(name, config[name], minimum=0.0, maximum=1.0, upper_inclusive=False)
    max_tokens = int(config["memory_max_tokens"])
    topk = int(config["memory_topk"])
    if not 2 <= topk <= max_tokens:
        raise ValueError("memory_topk must satisfy 2 <= memory_topk <= memory_max_tokens; "
                         f"received memory_topk={config['memory_topk']!r}, memory_max_tokens={config['memory_max_tokens']!r}")
    base_channels, heads = int(config["base_channels"]), int(config["decoder_num_heads"])
    widths = (base_channels, base_channels * 2, base_channels * 3, base_channels * 4)
    if any(width % heads for width in widths):
        raise ValueError("all decoder scale widths must be divisible by decoder_num_heads; "
                         f"decoder_num_heads={heads}, widths={widths}")
    return config
