import json

import pytest
import torch

from tools.inspect_legacy_checkpoint import compare_state_dicts, extract_state_dict


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
