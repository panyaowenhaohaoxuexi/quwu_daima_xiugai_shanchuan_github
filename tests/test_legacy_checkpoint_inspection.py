import json
from collections import UserDict

import pytest
import torch

from tools.inspect_legacy_checkpoint import compare_state_dicts, extract_state_dict
from utils.model_config_validation import MODEL_CONFIG_KEYS


def _model_config():
    return {
        "base_channels": 8, "router_hidden_channels": 8, "deform_num_samples": 4,
        "deform_max_offset": 2.0, "num_structure_renderers": 2, "memory_max_tokens": 16,
        "memory_topk": 2, "memory_query_chunk_size": 64, "memory_attention_temperature": 0.07,
        "memory_reliability_epsilon": 1e-6, "memory_reliable_ratio_threshold": 0.01,
        "memory_confidence_threshold": 0.1, "memory_exclusion_extra_margin": 0,
        "boundary_width": 1, "decoder_num_heads": 4, "decoder_depth": 1,
        "decoder_window_size": 7, "decoder_window_chunk_size": 128, "decoder_mlp_ratio": 4.0,
        "decoder_attention_dropout": 0.0, "decoder_projection_dropout": 0.0,
        "decoder_ffn_dropout": 0.0,
    }


def test_legacy_inspection_reports_missing_unexpected_and_shape_mismatch_without_inference():
    checkpoint = {"model": {"same": torch.zeros(2), "wrong": torch.zeros(3), "extra": torch.zeros(1)}}
    expected = {"same": torch.zeros(2), "wrong": torch.zeros(2), "missing": torch.zeros(1)}

    report = compare_state_dicts(extract_state_dict(checkpoint), expected)

    assert report["missing_keys"] == ["missing"]
    assert report["unexpected_keys"] == ["extra"]
    assert report["shape_mismatches"] == [{"key": "wrong", "checkpoint": [3], "expected": [2]}]


def test_legacy_inspection_reports_v2_compatibility_without_rejecting_old_weights(tmp_path, capsys):
    from tools.inspect_legacy_checkpoint import main

    path = tmp_path / "legacy.pt"
    torch.save({"model": {"weight": torch.ones(1)}}, path)
    report = main(["--checkpoint", str(path)])
    assert report["format_version"] is None
    assert not report["has_v2_format"]
    assert not report["is_formal_v2_compatible"]
    assert json.loads(capsys.readouterr().out)["checkpoint_key_count"] == 1


def test_legacy_static_v2_schema_rejects_empty_or_non_tensor_required_states(tmp_path, capsys):
    from tools.inspect_legacy_checkpoint import main

    base = {
        "format_version": 2, "training_stage": "ema",
        "config": {}, "model": {"legacy_weight": torch.ones(1)},
        "student": {}, "teacher": {"weight": "not-a-tensor"},
    }
    path = tmp_path / "invalid-v2.pt"
    torch.save(base, path)
    report = main(["--checkpoint", str(path)])
    assert not report["has_expected_state_dict_fields"]
    assert not report["is_formal_v2_compatible"]
    assert json.loads(capsys.readouterr().out)["checkpoint_key_count"] == 1


@pytest.mark.parametrize(
    ("key", "value"),
    (("decoder_depth", 0.5), ("decoder_window_size", 0),
     ("decoder_attention_dropout", 1.0), ("memory_topk", 17),
     ("decoder_num_heads", 5)),
)
def test_legacy_static_v2_report_rejects_invalid_model_config_values(tmp_path, capsys, key, value):
    from tools.inspect_legacy_checkpoint import main

    config = _model_config()
    config[key] = value
    path = tmp_path / f"invalid-{key}.pt"
    torch.save({"format_version": 2, "training_stage": "source", "config": config,
                "model": {"weight": torch.ones(1)}}, path)
    report = main(["--checkpoint", str(path)])
    assert not report["has_valid_model_config"]
    assert not report["is_formal_v2_compatible"]
    assert json.loads(capsys.readouterr().out)["has_complete_model_config"]


def test_legacy_static_v2_report_accepts_complete_valid_config_without_model_loading(tmp_path):
    from tools.inspect_legacy_checkpoint import main

    path = tmp_path / "static-v2.pt"
    torch.save({"format_version": 2, "training_stage": "source", "config": _model_config(),
                "model": {"weight": torch.ones(1)}}, path)
    report = main(["--checkpoint", str(path)])
    assert set(MODEL_CONFIG_KEYS) <= set(_model_config())
    assert report["has_valid_model_config"]
    assert report["is_formal_v2_compatible"]


def test_legacy_static_v2_report_accepts_nonempty_mapping_state_dicts(tmp_path):
    from tools.inspect_legacy_checkpoint import main

    path = tmp_path / "mapping-v2.pt"
    torch.save({"format_version": 2, "training_stage": "source", "config": _model_config(),
                "model": UserDict(weight=torch.ones(1))}, path)
    report = main(["--checkpoint", str(path)])
    assert report["has_expected_state_dict_fields"]
    assert report["is_formal_v2_compatible"]
