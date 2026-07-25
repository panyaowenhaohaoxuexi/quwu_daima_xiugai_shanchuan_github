import random
import copy

import numpy as np
import pytest
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


def test_source_checkpoint_config_persists_every_actual_loss_weight_without_private_cli_state():
    from option.Teacher import LOSS_WEIGHT_NAMES, build_parser, validate_config
    from option._formal_config import persisted_config_from_args

    expected = {
        "q_l1_weight": 1.1,
        "q_gradient_weight": 1.2,
        "q_ssim_weight": 1.3,
        "rec_l1_weight": 1.4,
        "rec_gradient_weight": 1.5,
        "rec_ssim_weight": 1.6,
        "boundary_l1_weight": 1.7,
        "boundary_gradient_weight": 1.8,
    }
    argv = [argument for name, value in expected.items() for argument in (f"--{name}", str(value))]
    args = validate_config(build_parser().parse_args(argv))
    checkpoint = build_source_checkpoint(
        {}, {}, None, 0, 0, persisted_config_from_args(args), "transmission", {},
    )

    assert {name: checkpoint["config"][name] for name in LOSS_WEIGHT_NAMES} == expected
    assert not any(key.startswith("_") for key in checkpoint["config"])


def test_ema_checkpoint_config_uses_current_formal_loss_weights_not_source_values():
    from EMA import build_ema_checkpoint_config
    from option.EMA import build_parser, validate_config
    from option.Teacher import LOSS_WEIGHT_NAMES

    args = validate_config(build_parser().parse_args(["--formal_training"]))
    source_model_config = {name: 9.1 + index / 10 for index, name in enumerate(LOSS_WEIGHT_NAMES)}
    source_model_config["formal_training"] = False
    source_model_config["_legacy_private"] = "discard"

    config = build_ema_checkpoint_config(source_model_config, args)

    assert config["formal_training"] is True
    assert {name: config[name] for name in LOSS_WEIGHT_NAMES} == {
        name: getattr(args, name) for name in LOSS_WEIGHT_NAMES
    }
    assert not any(key.startswith("_") for key in config)


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
    source_loader_generator = torch.Generator().manual_seed(31)
    checkpoint = build_source_checkpoint(
        model.state_dict(), optimizer.state_dict(), None, 7, 2,
        {"base_channels": 8}, "transmission", capture_rng_state(
            dataloader_generators={"source": source_loader_generator},
        ),
        sampler_states={"source": sampler.state_dict()},
        empty_omega_streaks={"source": 5},
    )
    resumed_model = nn.Linear(1, 1)
    resumed_optimizer = torch.optim.SGD(resumed_model.parameters(), lr=0.1)
    resumed_sampler = StatefulRandomSampler(3, seed=99)
    resumed_loader_generator = torch.Generator().manual_seed(99)

    restored = restore_source_training_state(
        checkpoint, resumed_model, resumed_optimizer, resumed_sampler, "transmission",
        dataloader_generators={"source": resumed_loader_generator},
    )

    assert restored == {"global_step": 7, "epoch": 2, "empty_omega_streak": 5}
    assert resumed_sampler.next_sample_position == 1
    assert torch.equal(resumed_model.weight, model.weight)
    assert torch.equal(resumed_loader_generator.get_state(), source_loader_generator.get_state())


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
    real_loader_generator = torch.Generator().manual_seed(41)
    source_anchor_loader_generator = torch.Generator().manual_seed(42)
    checkpoint = build_ema_checkpoint(
        student.state_dict(), teacher.state_dict(), optimizer.state_dict(), None, 5, 6, 3,
        {"base_channels": 8}, "transmission", capture_rng_state(
            dataloader_generators={
                "real": real_loader_generator,
                "source_anchor": source_anchor_loader_generator,
            },
        ),
        sampler_states={"source": source_sampler.state_dict(), "real": real_sampler.state_dict()},
        empty_omega_streaks={"source": 1, "anchor": 2},
    )
    new_student, new_teacher = nn.Linear(1, 1), nn.Linear(1, 1)
    new_optimizer = torch.optim.SGD(new_student.parameters(), lr=0.1)
    new_source, new_real = StatefulRandomSampler(3, 9), StatefulRandomSampler(4, 9)
    resumed_real_generator = torch.Generator().manual_seed(99)
    resumed_anchor_generator = torch.Generator().manual_seed(98)

    restored = restore_ema_training_state(
        checkpoint, new_student, new_teacher, new_optimizer, new_source, new_real, "transmission",
        dataloader_generators={
            "real": resumed_real_generator,
            "source_anchor": resumed_anchor_generator,
        },
    )

    assert restored["source_global_step"] == 5 and restored["ema_global_step"] == 6
    assert new_source.next_sample_position == 1 and new_real.next_sample_position == 2
    assert torch.equal(new_teacher.weight, teacher.weight)
    assert torch.equal(resumed_real_generator.get_state(), real_loader_generator.get_state())
    assert torch.equal(resumed_anchor_generator.get_state(), source_anchor_loader_generator.get_state())


def test_source_restore_rejects_missing_loader_rng_before_any_state_write():
    from torch import nn
    from data import StatefulRandomSampler
    from training.checkpointing import restore_source_training_state

    model = nn.BatchNorm1d(2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    source_sampler = StatefulRandomSampler(3, seed=1)
    checkpoint = build_source_checkpoint(
        model.state_dict(), optimizer.state_dict(), None, 1, 0,
        {"base_channels": 8}, "transmission", capture_rng_state(),
        sampler_states={"source": source_sampler.state_dict()},
    )
    checkpoint["rng_state"]["dataloader_generators"] = {}
    resumed_model = nn.BatchNorm1d(2)
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=0.2)
    resumed_sampler = StatefulRandomSampler(3, seed=9)
    resumed_generator = torch.Generator().manual_seed(10)
    before = {
        "model": copy.deepcopy(resumed_model.state_dict()),
        "optimizer": copy.deepcopy(resumed_optimizer.state_dict()),
        "sampler": resumed_sampler.state_dict(),
        "generator": resumed_generator.get_state().clone(),
    }

    with pytest.raises(ValueError, match=r"source.*DataLoader generator.*source"):
        restore_source_training_state(
            checkpoint, resumed_model, resumed_optimizer, resumed_sampler, "transmission",
            dataloader_generators={"source": resumed_generator},
        )

    _assert_nested_equal(resumed_model.state_dict(), before["model"])
    _assert_nested_equal(resumed_optimizer.state_dict(), before["optimizer"])
    assert resumed_sampler.state_dict() == before["sampler"]
    assert torch.equal(resumed_generator.get_state(), before["generator"])


@pytest.mark.parametrize(
    ("stored_generators", "missing"),
    [({"source_anchor": torch.Generator().manual_seed(1).get_state()}, "real"),
     ({"real": torch.Generator().manual_seed(2).get_state()}, "source_anchor")],
)
def test_ema_restore_rejects_each_missing_required_loader_rng(stored_generators, missing):
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
    )
    checkpoint["rng_state"]["dataloader_generators"] = stored_generators

    with pytest.raises(ValueError, match=missing):
        restore_ema_training_state(
            checkpoint, nn.Linear(1, 1), nn.Linear(1, 1),
            torch.optim.SGD(nn.Linear(1, 1).parameters(), lr=0.1),
            StatefulRandomSampler(2, 3), StatefulRandomSampler(2, 4), "transmission",
            dataloader_generators={
                "real": torch.Generator().manual_seed(3),
                "source_anchor": torch.Generator().manual_seed(4),
            },
        )


def test_ema_restore_rejects_missing_loader_rng_before_any_state_write():
    from torch import nn
    from data import StatefulRandomSampler
    from training.checkpointing import restore_ema_training_state

    student, teacher = nn.BatchNorm1d(2), nn.BatchNorm1d(2)
    optimizer = torch.optim.Adam(student.parameters(), lr=0.1)
    source_sampler, real_sampler = StatefulRandomSampler(3, 1), StatefulRandomSampler(3, 2)
    checkpoint = build_ema_checkpoint(
        student.state_dict(), teacher.state_dict(), optimizer.state_dict(), None, 0, 0, 0,
        {"base_channels": 8}, "transmission", capture_rng_state(),
        sampler_states={"source": source_sampler.state_dict(), "real": real_sampler.state_dict()},
    )
    checkpoint["rng_state"]["dataloader_generators"] = {}
    resumed_student, resumed_teacher = nn.BatchNorm1d(2), nn.BatchNorm1d(2)
    resumed_optimizer = torch.optim.Adam(resumed_student.parameters(), lr=0.2)
    resumed_source, resumed_real = StatefulRandomSampler(3, 9), StatefulRandomSampler(3, 10)
    resumed_real_generator = torch.Generator().manual_seed(11)
    resumed_anchor_generator = torch.Generator().manual_seed(12)
    before = {
        "student": copy.deepcopy(resumed_student.state_dict()),
        "teacher": copy.deepcopy(resumed_teacher.state_dict()),
        "optimizer": copy.deepcopy(resumed_optimizer.state_dict()),
        "source_sampler": resumed_source.state_dict(),
        "real_sampler": resumed_real.state_dict(),
        "real_generator": resumed_real_generator.get_state().clone(),
        "anchor_generator": resumed_anchor_generator.get_state().clone(),
    }

    with pytest.raises(ValueError, match=r"ema.*DataLoader generator.*real"):
        restore_ema_training_state(
            checkpoint, resumed_student, resumed_teacher, resumed_optimizer,
            resumed_source, resumed_real, "transmission",
            dataloader_generators={
                "real": resumed_real_generator,
                "source_anchor": resumed_anchor_generator,
            },
        )

    _assert_nested_equal(resumed_student.state_dict(), before["student"])
    _assert_nested_equal(resumed_teacher.state_dict(), before["teacher"])
    _assert_nested_equal(resumed_optimizer.state_dict(), before["optimizer"])
    assert resumed_source.state_dict() == before["source_sampler"]
    assert resumed_real.state_dict() == before["real_sampler"]
    assert torch.equal(resumed_real_generator.get_state(), before["real_generator"])
    assert torch.equal(resumed_anchor_generator.get_state(), before["anchor_generator"])


@pytest.mark.parametrize("stage", ["source", "ema"])
def test_restore_rejects_missing_loader_rng_mapping(stage):
    from torch import nn
    from data import StatefulRandomSampler
    from training.checkpointing import restore_ema_training_state, restore_source_training_state

    if stage == "source":
        model = nn.Linear(1, 1)
        checkpoint = build_source_checkpoint(
            model.state_dict(), torch.optim.SGD(model.parameters(), lr=0.1).state_dict(), None, 0, 0,
            {"base_channels": 8}, "transmission", capture_rng_state(),
            sampler_states={"source": StatefulRandomSampler(2, 1).state_dict()},
        )
        del checkpoint["rng_state"]["dataloader_generators"]
        restore = lambda: restore_source_training_state(
            checkpoint, nn.Linear(1, 1), torch.optim.SGD(nn.Linear(1, 1).parameters(), lr=0.1),
            StatefulRandomSampler(2, 2), "transmission",
            dataloader_generators={"source": torch.Generator().manual_seed(3)},
        )
    else:
        student, teacher = nn.Linear(1, 1), nn.Linear(1, 1)
        checkpoint = build_ema_checkpoint(
            student.state_dict(), teacher.state_dict(), torch.optim.SGD(student.parameters(), lr=0.1).state_dict(),
            None, 0, 0, 0, {"base_channels": 8}, "transmission", capture_rng_state(),
            sampler_states={
                "source": StatefulRandomSampler(2, 1).state_dict(),
                "real": StatefulRandomSampler(2, 2).state_dict(),
            },
        )
        del checkpoint["rng_state"]["dataloader_generators"]
        restore = lambda: restore_ema_training_state(
            checkpoint, nn.Linear(1, 1), nn.Linear(1, 1),
            torch.optim.SGD(nn.Linear(1, 1).parameters(), lr=0.1),
            StatefulRandomSampler(2, 3), StatefulRandomSampler(2, 4), "transmission",
            dataloader_generators={
                "real": torch.Generator().manual_seed(3),
                "source_anchor": torch.Generator().manual_seed(4),
            },
        )

    with pytest.raises(ValueError, match=rf"{stage}.*DataLoader generator"):
        restore()


def test_source_restore_rejects_non_tensor_loader_rng_state():
    from torch import nn
    from data import StatefulRandomSampler
    from training.checkpointing import restore_source_training_state

    model = nn.Linear(1, 1)
    checkpoint = build_source_checkpoint(
        model.state_dict(), torch.optim.SGD(model.parameters(), lr=0.1).state_dict(), None, 0, 0,
        {"base_channels": 8}, "transmission", capture_rng_state(),
        sampler_states={"source": StatefulRandomSampler(2, 1).state_dict()},
    )
    checkpoint["rng_state"]["dataloader_generators"] = {"source": "invalid"}

    with pytest.raises(ValueError, match=r"source.*DataLoader generator state.*torch tensor"):
        restore_source_training_state(
            checkpoint, nn.Linear(1, 1), torch.optim.SGD(nn.Linear(1, 1).parameters(), lr=0.1),
            StatefulRandomSampler(2, 2), "transmission",
            dataloader_generators={"source": torch.Generator().manual_seed(3)},
        )


def _assert_nested_equal(actual, expected):
    if torch.is_tensor(actual):
        assert torch.equal(actual, expected)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for value, expected_value in zip(actual, expected):
            _assert_nested_equal(value, expected_value)
    else:
        assert actual == expected


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
        "tir_dataset_percentile_low_value": 100.0, "tir_dataset_percentile_high_value": 900.0,
        "tir_channel_tolerance_float": 1e-4,
    }
    restored = tir_normalization_config_from_checkpoint(config)

    assert restored["normalization"] == "percentile"
    assert restored["percentile_scope"] == "dataset"
    assert restored["channel_tolerance_code_values"] == 2
    assert restored["dataset_percentile_high_value"] == 900.0
