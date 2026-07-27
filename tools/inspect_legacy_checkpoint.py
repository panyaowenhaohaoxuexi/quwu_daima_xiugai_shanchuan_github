"""Read-only legacy checkpoint inspection; never part of formal inference."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping

import torch

from utils.checkpoint import CHECKPOINT_FORMAT_VERSION, MODEL_CONFIG_KEYS, require_complete_model_config


def _is_nonempty_tensor_state(value):
    return (isinstance(value, Mapping) and bool(value) and all(
        isinstance(key, str) and torch.is_tensor(tensor) for key, tensor in value.items()
    ))


def _has_valid_model_config(config):
    """Static schema validation only; this tool never constructs or loads a model."""
    try:
        require_complete_model_config(config)
    except (TypeError, ValueError):
        return False
    return True


def extract_state_dict(checkpoint):
    """Extract one explicit state mapping without guessing formal load behavior."""
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a dictionary")
    for key in ("model", "state_dict", "student", "teacher"):
        value = checkpoint.get(key)
        if _is_nonempty_tensor_state(value):
            return value
    if _is_nonempty_tensor_state(checkpoint):
        return checkpoint
    raise ValueError("checkpoint contains no recognizable tensor state dictionary")


def compare_state_dicts(checkpoint_state, expected_state):
    checkpoint_keys, expected_keys = set(checkpoint_state), set(expected_state)
    missing = sorted(expected_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - expected_keys)
    mismatches = []
    for key in sorted(checkpoint_keys & expected_keys):
        actual_shape, expected_shape = list(checkpoint_state[key].shape), list(expected_state[key].shape)
        if actual_shape != expected_shape:
            mismatches.append({"key": key, "checkpoint": actual_shape, "expected": expected_shape})
    return {"missing_keys": missing, "unexpected_keys": unexpected, "shape_mismatches": mismatches}


def build_parser():
    parser = argparse.ArgumentParser("legacy-checkpoint-inspection")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--reference_checkpoint")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    state = extract_state_dict(checkpoint)
    stage = checkpoint.get("training_stage") if isinstance(checkpoint, dict) else None
    config = checkpoint.get("config") if isinstance(checkpoint, dict) else None
    expected_states = ((stage == "source" and "model" in checkpoint) or
                       (stage == "ema" and "student" in checkpoint and "teacher" in checkpoint)) if isinstance(checkpoint, dict) else False
    state_fields_are_tensors = all(
        _is_nonempty_tensor_state(checkpoint.get(key))
        for key in (("model",) if stage == "source" else ("student", "teacher") if stage == "ema" else ())
    )
    report = {"checkpoint_key_count": len(state), "keys": sorted(state),
              "format_version": checkpoint.get("format_version") if isinstance(checkpoint, dict) else None,
              "has_v2_format": isinstance(checkpoint, dict) and checkpoint.get("format_version") == CHECKPOINT_FORMAT_VERSION,
              "has_valid_training_stage": stage in ("source", "ema"),
              "has_complete_model_config": isinstance(config, Mapping) and all(key in config for key in MODEL_CONFIG_KEYS),
              "has_valid_model_config": _has_valid_model_config(config),
              "has_expected_state_dict_fields": expected_states and state_fields_are_tensors}
    report["is_formal_v2_compatible"] = all((report["has_v2_format"], report["has_valid_training_stage"],
                                              report["has_complete_model_config"], report["has_valid_model_config"],
                                              report["has_expected_state_dict_fields"]))
    if args.reference_checkpoint:
        reference = extract_state_dict(torch.load(args.reference_checkpoint, map_location="cpu"))
        report.update(compare_state_dicts(state, reference))
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
