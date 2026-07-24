"""Single-process sampler with explicit committed cursor for checkpoint recovery."""

import hashlib
import json
import os

import torch
from torch.utils.data import Sampler


def validate_single_process_world():
    """Reject distributed execution for the first exact-resume implementation."""
    raw_world_size = os.environ.get("WORLD_SIZE", "1")
    try:
        world_size = int(raw_world_size)
    except ValueError as error:
        raise ValueError(f"WORLD_SIZE must be an integer, got {raw_world_size!r}") from error
    if world_size != 1:
        raise ValueError(
            "exact resumable training currently requires WORLD_SIZE == 1; "
            f"got WORLD_SIZE={world_size}"
        )


def manifest_fingerprint(records):
    """Stable fingerprint for a dataset manifest independent of dict key order."""
    canonical = sorted(json.dumps(record, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
                       for record in records)
    return hashlib.sha256("\n".join(canonical).encode("utf-8")).hexdigest()


class StatefulRandomSampler(Sampler):
    def __init__(self, data_source_size, seed=0):
        if data_source_size < 0:
            raise ValueError("data_source_size must be non-negative")
        self.data_source_size = int(data_source_size)
        self.seed = int(seed)
        self.epoch = 0
        self.permutation = self._make_permutation()
        self.next_sample_position = 0

    def _make_permutation(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return torch.randperm(self.data_source_size, generator=generator).tolist()

    def __iter__(self):
        return iter(self.permutation[self.next_sample_position:])

    def __len__(self):
        return self.data_source_size - self.next_sample_position

    def commit(self, sample_count):
        if sample_count < 0:
            raise ValueError("sample_count must be non-negative")
        self.next_sample_position = min(self.data_source_size, self.next_sample_position + int(sample_count))

    def advance_epoch(self):
        self.epoch += 1
        self.permutation = self._make_permutation()
        self.next_sample_position = 0

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "seed": self.seed,
            "permutation": list(self.permutation),
            "next_sample_position": self.next_sample_position,
        }

    def load_state_dict(self, state):
        required = {"epoch", "seed", "permutation", "next_sample_position"}
        if set(state) != required:
            raise ValueError(f"invalid sampler state keys: {set(state)}")
        if len(state["permutation"]) != self.data_source_size:
            raise ValueError("sampler state dataset size mismatch")
        self.epoch = int(state["epoch"])
        self.seed = int(state["seed"])
        self.permutation = [int(value) for value in state["permutation"]]
        self.next_sample_position = int(state["next_sample_position"])
        if not 0 <= self.next_sample_position <= self.data_source_size:
            raise ValueError("invalid next_sample_position")
