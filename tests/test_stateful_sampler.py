import pytest

from data.stateful_sampler import (
    StatefulRandomSampler,
    manifest_fingerprint,
    validate_single_process_world,
)


def test_stateful_sampler_restores_saved_permutation_from_next_position():
    sampler = StatefulRandomSampler(8, seed=42)
    iterator = iter(sampler)
    consumed = [next(iterator) for _ in range(3)]
    sampler.commit(3)
    state = sampler.state_dict()

    resumed = StatefulRandomSampler(8, seed=999)
    resumed.load_state_dict(state)

    assert len(set(consumed)) == 3
    assert list(resumed) == state["permutation"][3:]


def test_manifest_fingerprint_is_stable_and_detects_path_changes():
    first = manifest_fingerprint([{"sample": "a", "density": "d1"}, {"sample": "b", "density": "d2"}])
    same = manifest_fingerprint([{"density": "d2", "sample": "b"}, {"density": "d1", "sample": "a"}])
    changed = manifest_fingerprint([{"sample": "a", "density": "changed"}, {"sample": "b", "density": "d2"}])

    assert first == same
    assert first != changed


def test_single_process_guard_rejects_distributed_world(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")

    with pytest.raises(ValueError, match="WORLD_SIZE == 1"):
        validate_single_process_world()
