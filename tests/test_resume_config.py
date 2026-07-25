import pytest

from training.resume_config import (
    apply_ema_resume_config,
    apply_source_resume_config,
    validate_source_resume_semantics,
)


OBJECTIVE_KEYS = (
    "formal_training", "q_l1_weight", "q_gradient_weight", "q_ssim_weight",
    "rec_l1_weight", "rec_gradient_weight", "rec_ssim_weight",
    "boundary_l1_weight", "boundary_gradient_weight",
)


def _objective_config(*, formal_training, start):
    return {
        "formal_training": formal_training,
        "q_l1_weight": start + 0.0,
        "q_gradient_weight": start + 0.1,
        "q_ssim_weight": start + 0.2,
        "rec_l1_weight": start + 0.3,
        "rec_gradient_weight": start + 0.4,
        "rec_ssim_weight": start + 0.5,
        "boundary_l1_weight": start + 0.6,
        "boundary_gradient_weight": start + 0.7,
    }


def test_ema_resume_restores_checkpoint_adaptation_values_by_default():
    current = {
        "base_channels": 16, "tir_normalization": "dtype_range", "density_gt_semantics": "transmission",
        "route_tau_end": 0.2, "train_size": 256, "ema_decay": 0.9, "lambda_anchor": 1.0, "epochs": 2,
        "train_data_dir": "new-source", "real_data_dir": "new-real",
    }
    saved = {
        "base_channels": 8, "tir_normalization": "fixed_range", "density_gt_semantics": "density",
        "route_tau_end": 0.4, "train_size": 32, "ema_decay": 0.99, "lambda_anchor": 0.3, "epochs": 9,
        "train_data_dir": "old-source", "real_data_dir": "old-real",
    }

    restored, diff = apply_ema_resume_config(current, saved, allow_training_override=False)

    assert restored["base_channels"] == 8
    assert restored["tir_normalization"] == "fixed_range"
    assert restored["train_size"] == 32
    assert restored["ema_decay"] == 0.99
    assert restored["lambda_anchor"] == 0.3
    assert restored["epochs"] == 2
    assert restored["train_data_dir"] == "new-source"
    assert restored["real_data_dir"] == "new-real"
    assert diff == {}


def test_ema_resume_keeps_only_explicitly_permitted_training_overrides():
    current = {
        "base_channels": 16, "tir_normalization": "dtype_range", "density_gt_semantics": "transmission",
        "route_tau_end": 0.2, "train_size": 256, "ema_decay": 0.9, "lambda_anchor": 1.0, "epochs": 2,
        "train_data_dir": "new-source", "real_data_dir": "new-real",
    }
    saved = {
        "base_channels": 8, "tir_normalization": "fixed_range", "density_gt_semantics": "density",
        "route_tau_end": 0.4, "train_size": 32, "ema_decay": 0.99, "lambda_anchor": 0.3, "epochs": 9,
        "train_data_dir": "old-source", "real_data_dir": "old-real",
    }

    restored, diff = apply_ema_resume_config(current, saved, allow_training_override=True)

    assert restored["base_channels"] == 8  # semantic model key cannot change
    assert restored["tir_normalization"] == "fixed_range"
    assert restored["train_size"] == 32
    assert restored["ema_decay"] == 0.9
    assert restored["lambda_anchor"] == 1.0
    assert restored["train_data_dir"] == "new-source"
    assert restored["real_data_dir"] == "new-real"
    assert diff == {"ema_decay": (0.99, 0.9), "lambda_anchor": (0.3, 1.0)}


def test_source_resume_rejects_changed_preprocessing_semantics():
    stored = {
        "base_channels": 8,
        "tir_normalization": "dtype_range",
        "density_gt_semantics": "transmission",
        "pair_alignment_policy": "strict",
        "train_size": 32,
    }
    current = dict(stored, tir_normalization="fixed_range")

    with pytest.raises(ValueError, match="source resume semantic configuration mismatch"):
        validate_source_resume_semantics(stored, current)


def test_source_resume_locks_all_training_objectives_from_checkpoint_by_default():
    checkpoint = _objective_config(formal_training=True, start=1.1)
    current = _objective_config(formal_training=False, start=7.1)

    resolved, diff = apply_source_resume_config(
        current, checkpoint, allow_training_override=False,
        explicit_objective_keys=set(OBJECTIVE_KEYS),
    )

    assert {key: resolved[key] for key in OBJECTIVE_KEYS} == checkpoint
    assert diff == {}


def test_source_resume_override_changes_only_explicit_training_objectives():
    checkpoint = _objective_config(formal_training=True, start=1.1)
    current = _objective_config(formal_training=True, start=7.1)
    explicit = {"q_gradient_weight", "rec_ssim_weight"}

    resolved, diff = apply_source_resume_config(
        current, checkpoint, allow_training_override=True,
        explicit_objective_keys=explicit,
    )

    assert resolved["q_gradient_weight"] == current["q_gradient_weight"]
    assert resolved["rec_ssim_weight"] == current["rec_ssim_weight"]
    for key in set(OBJECTIVE_KEYS).difference(explicit):
        assert resolved[key] == checkpoint[key]
    assert diff == {
        "q_gradient_weight": (checkpoint["q_gradient_weight"], current["q_gradient_weight"]),
        "rec_ssim_weight": (checkpoint["rec_ssim_weight"], current["rec_ssim_weight"]),
    }


def test_source_resume_override_without_explicit_objectives_keeps_checkpoint_values():
    checkpoint = _objective_config(formal_training=True, start=1.1)
    current = _objective_config(formal_training=False, start=7.1)

    resolved, diff = apply_source_resume_config(
        current, checkpoint, allow_training_override=True, explicit_objective_keys=set(),
    )

    assert {key: resolved[key] for key in OBJECTIVE_KEYS} == checkpoint
    assert diff == {}


def test_source_resume_allows_explicit_formal_training_true_override_only():
    checkpoint = _objective_config(formal_training=False, start=1.1)
    current = _objective_config(formal_training=True, start=1.1)

    resolved, diff = apply_source_resume_config(
        current, checkpoint, allow_training_override=True,
        explicit_objective_keys={"formal_training"},
    )

    assert resolved["formal_training"] is True
    assert diff == {"formal_training": (False, True)}


@pytest.mark.parametrize("missing", ("formal_training", "q_gradient_weight"))
def test_source_resume_rejects_checkpoint_missing_training_objective(missing):
    checkpoint = _objective_config(formal_training=True, start=1.1)
    checkpoint.pop(missing)

    with pytest.raises(ValueError, match=rf"training objective configuration.*{missing}"):
        apply_source_resume_config(
            _objective_config(formal_training=False, start=7.1), checkpoint,
            allow_training_override=False, explicit_objective_keys=set(),
        )


@pytest.mark.parametrize("allow_training_override", (False, True))
def test_source_resume_override_never_bypasses_semantic_validation(allow_training_override):
    checkpoint = dict(_objective_config(formal_training=True, start=1.1), base_channels=8)
    current = dict(_objective_config(formal_training=False, start=7.1), base_channels=16)

    resolved, _ = apply_source_resume_config(
        current, checkpoint, allow_training_override=allow_training_override,
        explicit_objective_keys={"q_l1_weight"},
    )

    with pytest.raises(ValueError, match="source resume semantic configuration mismatch"):
        validate_source_resume_semantics(checkpoint, resolved)


def test_source_resume_persisted_config_keeps_checkpoint_objective_and_drops_private_fields():
    from argparse import Namespace
    from option._formal_config import persisted_config_from_args

    checkpoint = _objective_config(formal_training=True, start=1.1)
    resolved, _ = apply_source_resume_config(
        _objective_config(formal_training=False, start=7.1), checkpoint,
        allow_training_override=False, explicit_objective_keys=set(),
    )
    resolved["_parser_private"] = "discard"
    persisted = persisted_config_from_args(Namespace(**resolved))

    assert {key: persisted[key] for key in OBJECTIVE_KEYS} == checkpoint
    assert not any(key.startswith("_") for key in persisted)
