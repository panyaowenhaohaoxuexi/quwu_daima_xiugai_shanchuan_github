import random

import numpy as np
import torch

from training.checkpointing import (
    build_ema_checkpoint,
    build_source_checkpoint,
    capture_rng_state,
    restore_rng_state,
    validate_checkpoint_metadata,
    build_formal_model_from_config,
    tir_normalization_config_from_checkpoint,
)


def test_rng_state_round_trip_restores_python_numpy_and_torch():
    random.seed(5)
    np.random.seed(5)
    torch.manual_seed(5)
    state = capture_rng_state()
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
    restore_rng_state(state)
    actual = (random.random(), float(np.random.rand()), float(torch.rand(())))

    assert actual == expected


def test_checkpoint_metadata_requires_matching_stage_class_and_density_semantics():
    checkpoint = {
        "format_version": 1,
        "training_stage": "source",
        "model_class": "FogRoutedRGBTIRDehazer",
        "density_gt_semantics": "transmission",
    }
    validate_checkpoint_metadata(checkpoint, "source", "transmission")


def test_source_and_ema_checkpoint_schemas_do_not_mix_model_fields():
    source = build_source_checkpoint({}, {}, None, 2, 1, {"base_channels": 8}, "transmission", {})
    ema = build_ema_checkpoint({}, {}, {}, None, 3, 4, 1, {"base_channels": 8}, "transmission", {})

    assert "model" in source and "student" not in source and "teacher" not in source
    assert {"student", "teacher", "source_global_step", "ema_global_step"}.issubset(ema)
    assert "model" not in ema


def test_checkpoint_persists_distinct_source_and_real_sampler_states():
    state = {"epoch": 0, "seed": 1, "permutation": [0, 1], "next_sample_position": 1}
    checkpoint = build_ema_checkpoint(
        {}, {}, {}, None, 3, 4, 1, {"base_channels": 8}, "transmission", {},
        sampler_states={"source": state, "real": state},
        empty_omega_streaks={"source": 2, "anchor": 3},
    )

    assert checkpoint["sampler_states"]["source"]["next_sample_position"] == 1
    assert checkpoint["empty_omega_streaks"] == {"source": 2, "anchor": 3}


def test_source_training_state_restore_loads_model_optimizer_sampler_and_counters():
    from torch import nn
    from data import StatefulRandomSampler
    from training.checkpointing import restore_source_training_state

    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    sampler = StatefulRandomSampler(3, seed=4)
    sampler.commit(1)
    checkpoint = build_source_checkpoint(
        model.state_dict(), optimizer.state_dict(), None, 7, 2,
        {"base_channels": 8}, "transmission", capture_rng_state(),
        sampler_states={"source": sampler.state_dict()},
        empty_omega_streaks={"source": 5},
    )
    resumed_model = nn.Linear(1, 1)
    resumed_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=0.1)
    resumed_sampler = StatefulRandomSampler(3, seed=99)

    restored = restore_source_training_state(
        checkpoint, resumed_model, resumed_optimizer, resumed_sampler, "transmission"
    )

    assert restored == {"global_step": 7, "epoch": 2, "empty_omega_streak": 5}
    assert resumed_sampler.next_sample_position == 1
    assert torch.equal(resumed_model.weight, model.weight)


def test_source_restore_rejects_changed_dataset_manifest_before_iterator_creation():
    """A mid-epoch resume must never silently continue on changed samples."""
    import pytest
    from torch import nn
    from data import StatefulRandomSampler
    from training.checkpointing import restore_source_training_state

    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    sampler = StatefulRandomSampler(2, seed=7)
    checkpoint = build_source_checkpoint(
        model.state_dict(), optimizer.state_dict(), None, 1, 0,
        {"base_channels": 8}, "transmission", capture_rng_state(),
        sampler_states={"source": sampler.state_dict()},
        manifest_fingerprints={"source": "manifest-before"},
    )

    with pytest.raises(ValueError, match="manifest fingerprint mismatch"):
        restore_source_training_state(
            checkpoint, nn.Linear(1, 1), torch.optim.SGD(nn.Linear(1, 1).parameters(), lr=0.1),
            StatefulRandomSampler(2, seed=7), "transmission",
            manifest_fingerprint="manifest-after",
        )


def test_ema_training_state_restore_loads_student_teacher_and_two_samplers():
    from torch import nn
    from data import StatefulRandomSampler
    from training.checkpointing import restore_ema_training_state

    student, teacher = nn.Linear(1, 1), nn.Linear(1, 1)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    source_sampler, real_sampler = StatefulRandomSampler(3, 1), StatefulRandomSampler(4, 2)
    source_sampler.commit(1)
    real_sampler.commit(2)
    checkpoint = build_ema_checkpoint(
        student.state_dict(), teacher.state_dict(), optimizer.state_dict(), None, 5, 6, 3,
        {"base_channels": 8}, "transmission", capture_rng_state(),
        sampler_states={"source": source_sampler.state_dict(), "real": real_sampler.state_dict()},
        empty_omega_streaks={"source": 1, "anchor": 2},
    )
    new_student, new_teacher = nn.Linear(1, 1), nn.Linear(1, 1)
    new_optimizer = torch.optim.SGD(new_student.parameters(), lr=0.1)
    new_source, new_real = StatefulRandomSampler(3, 9), StatefulRandomSampler(4, 9)

    restored = restore_ema_training_state(
        checkpoint, new_student, new_teacher, new_optimizer, new_source, new_real, "transmission"
    )

    assert restored["source_global_step"] == 5 and restored["ema_global_step"] == 6
    assert new_source.next_sample_position == 1 and new_real.next_sample_position == 2
    assert torch.equal(new_teacher.weight, teacher.weight)


def test_ema_restore_rejects_changed_source_or_real_manifest():
    import pytest
    from torch import nn
    from data import StatefulRandomSampler
    from training.checkpointing import restore_ema_training_state

    student, teacher = nn.Linear(1, 1), nn.Linear(1, 1)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    checkpoint = build_ema_checkpoint(
        student.state_dict(), teacher.state_dict(), optimizer.state_dict(), None, 0, 0, 0,
        {"base_channels": 8}, "transmission", capture_rng_state(),
        sampler_states={
            "source": StatefulRandomSampler(2, 1).state_dict(),
            "real": StatefulRandomSampler(2, 2).state_dict(),
        },
        manifest_fingerprints={"source": "source-a", "real": "real-a"},
    )

    with pytest.raises(ValueError, match="manifest fingerprint mismatch"):
        restore_ema_training_state(
            checkpoint, nn.Linear(1, 1), nn.Linear(1, 1),
            torch.optim.SGD(nn.Linear(1, 1).parameters(), lr=0.1),
            StatefulRandomSampler(2, 1), StatefulRandomSampler(2, 2), "transmission",
            manifest_fingerprints={"source": "source-b", "real": "real-a"},
        )


def test_model_is_constructed_from_checkpoint_semantic_configuration():
    config = {
        "base_channels": 8, "router_hidden_channels": 5,
        "deform_num_samples": 3, "deform_max_offset": 1.5,
        "num_structure_renderers": 3, "memory_max_tokens": 16,
        "memory_topk": 4, "memory_query_chunk_size": 8,
        "memory_attention_temperature": 0.2, "memory_reliability_epsilon": 1e-6,
        "memory_reliable_ratio_threshold": 0.1, "memory_confidence_threshold": 0.2,
        "memory_exclusion_extra_margin": 0,
        "boundary_width": 2,
    }
    model = build_formal_model_from_config(config)

    assert model.base_channels == 8
    assert model.router.raw_weight_in.numel() == 5
    assert model.selector["h2"].out_channels == 3


def test_semantic_configuration_mismatch_is_rejected_even_without_shape_change():
    from training.checkpointing import validate_model_semantics_config
    stored = {
        "base_channels": 8, "router_hidden_channels": 8, "deform_num_samples": 4,
        "deform_max_offset": 2.0, "num_structure_renderers": 2, "memory_max_tokens": 16,
        "memory_topk": 2, "memory_attention_temperature": 0.07,
        "memory_reliability_epsilon": 1e-6, "memory_reliable_ratio_threshold": 0.01,
        "memory_confidence_threshold": 0.1, "boundary_width": 1,
        "memory_exclusion_extra_margin": 0,
    }
    current = dict(stored, memory_attention_temperature=0.2)

    import pytest
    with pytest.raises(ValueError, match="semantic configuration mismatch"):
        validate_model_semantics_config(stored, current)


def test_tir_loader_configuration_is_recovered_from_checkpoint_semantics():
    config = {
        "tir_normalization": "percentile", "tir_fixed_min": None, "tir_fixed_max": None,
        "tir_percentile_low": 2.0, "tir_percentile_high": 98.0,
        "tir_percentile_scope": "dataset", "tir_channel_tolerance_code_values": 2,
        "tir_channel_tolerance_float": 1e-4,
    }
    restored = tir_normalization_config_from_checkpoint(config)

    assert restored["normalization"] == "percentile"
    assert restored["percentile_scope"] == "dataset"
    assert restored["channel_tolerance_code_values"] == 2
