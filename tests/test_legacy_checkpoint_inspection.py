import torch

from tools.inspect_legacy_checkpoint import compare_state_dicts, extract_state_dict


def test_legacy_inspection_reports_missing_unexpected_and_shape_mismatch_without_inference():
    checkpoint = {"model": {"same": torch.zeros(2), "wrong": torch.zeros(3), "extra": torch.zeros(1)}}
    expected = {"same": torch.zeros(2), "wrong": torch.zeros(2), "missing": torch.zeros(1)}

    report = compare_state_dicts(extract_state_dict(checkpoint), expected)

    assert report["missing_keys"] == ["missing"]
    assert report["unexpected_keys"] == ["extra"]
    assert report["shape_mismatches"] == [{"key": "wrong", "checkpoint": [3], "expected": [2]}]
