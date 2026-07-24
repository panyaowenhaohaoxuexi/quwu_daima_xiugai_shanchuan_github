"""Read-only legacy checkpoint inspection; never part of formal inference."""

from __future__ import annotations

import argparse
import json

import torch


def extract_state_dict(checkpoint):
    """Extract one explicit state mapping without guessing formal load behavior."""
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must be a dictionary")
    for key in ("model", "state_dict", "student", "teacher"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and all(torch.is_tensor(item) for item in value.values()):
            return value
    if checkpoint and all(torch.is_tensor(item) for item in checkpoint.values()):
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
    report = {"checkpoint_key_count": len(state), "keys": sorted(state)}
    if args.reference_checkpoint:
        reference = extract_state_dict(torch.load(args.reference_checkpoint, map_location="cpu"))
        report.update(compare_state_dicts(state, reference))
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
