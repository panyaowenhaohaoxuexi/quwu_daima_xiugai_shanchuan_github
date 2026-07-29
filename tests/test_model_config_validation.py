import numpy as np
import pytest
import torch

from utils.model_config_validation import (
    MODEL_CONFIG_KEYS,
    require_nonnegative_integer,
    require_positive_integer,
    validate_model_config_values,
)


def _config():
    return {
        "base_channels": 8, "router_hidden_channels": 8, "deform_num_samples": 4,
        "deform_max_offset": 2.0, "num_structure_renderers": 2,
        "memory_max_tokens": 16, "memory_topk": 2, "memory_query_chunk_size": 64,
        "memory_attention_temperature": 0.07, "memory_reliability_epsilon": 1e-6,
        "memory_reliable_ratio_threshold": 0.01, "memory_confidence_threshold": 0.1,
        "decoder_num_heads": 4, "decoder_depth": 1, "decoder_window_size": 7,
        "decoder_window_chunk_size": 128, "decoder_mlp_ratio": 4.0,
        "decoder_attention_dropout": 0.0, "decoder_projection_dropout": 0.0,
        "decoder_ffn_dropout": 0.0,
    }


@pytest.mark.parametrize("value", [True, 3.0, 0.5, float("nan"), float("inf"), "3", 0, -1])
def test_positive_integer_validation_rejects_non_integral_or_nonpositive_values(value):
    with pytest.raises(ValueError, match="decoder_window_size.*received"):
        require_positive_integer("decoder_window_size", value)


def test_integer_validators_accept_python_and_numpy_integrals_and_only_nonnegative_allows_zero():
    assert require_positive_integer("channels", 3) == 3
    assert require_positive_integer("channels", np.int64(3)) == 3


@pytest.mark.parametrize("key,value", [
    ("decoder_window_size", 0.5), ("decoder_window_chunk_size", False),
    ("decoder_depth", float("nan")), ("deform_max_offset", float("inf")),
    ("decoder_attention_dropout", 1.0), ("memory_reliable_ratio_threshold", float("nan")),
])
def test_model_config_validation_rejects_invalid_model_values_before_model_construction(key, value):
    config = _config()
    config[key] = value
    with pytest.raises(ValueError, match=key):
        validate_model_config_values(config)


def test_model_config_validation_has_exact_model_only_schema_and_accepts_complete_values():
    config = _config()
    assert set(config) == set(MODEL_CONFIG_KEYS)
    state_before = torch.get_rng_state()
    assert validate_model_config_values(config) is config
    assert torch.equal(state_before, torch.get_rng_state())
