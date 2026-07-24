import pytest

from training.resume_config import apply_ema_resume_config, validate_source_resume_semantics


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
