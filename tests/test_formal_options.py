import argparse
import importlib
import sys

import pytest


LOSS_WEIGHT_ARGUMENTS = (
    "q_l1_weight", "q_gradient_weight", "q_ssim_weight",
    "rec_l1_weight", "rec_gradient_weight", "rec_ssim_weight",
    "boundary_l1_weight", "boundary_gradient_weight",
)


FORMAL_LOSS_WEIGHTS = {
    "q_l1_weight": 1.0,
    "q_gradient_weight": 0.5,
    "q_ssim_weight": 0.5,
    "rec_l1_weight": 1.0,
    "rec_gradient_weight": 0.2,
    "rec_ssim_weight": 0.2,
    "boundary_l1_weight": 1.0,
    "boundary_gradient_weight": 0.5,
}


def test_option_modules_are_pure_on_import_and_validate_formal_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    original = list(sys.argv)
    sys.argv = ["pytest", "--unexpected-pytest-argument"]
    try:
        teacher = importlib.reload(importlib.import_module("option.Teacher"))
        ema = importlib.reload(importlib.import_module("option.EMA"))
    finally:
        sys.argv = original

    assert not list(tmp_path.iterdir())
    teacher.validate_config(teacher.build_parser().parse_args([]))
    source_config = teacher.persisted_config_from_args(teacher.build_parser().parse_args([]))
    resolved, _ = ema.resolve_ema_config(ema.build_parser().parse_args([]), source_config, allow_training_override=False)
    ema.validate_config(argparse.Namespace(**resolved))


def test_tir_loader_mapping_keeps_all_persisted_preprocessing_semantics():
    from option.Teacher import build_parser, tir_normalization_config_from_args

    args = build_parser().parse_args([
        "--tir_normalization", "percentile", "--tir_percentile_scope", "dataset",
        "--tir_percentile_low", "2", "--tir_percentile_high", "98",
        "--tir_dataset_percentile_low_value", "100", "--tir_dataset_percentile_high_value", "900",
        "--tir_channel_tolerance_code_values", "2", "--tir_channel_tolerance_float", "0.0001",
    ])
    config = tir_normalization_config_from_args(args)
    assert config["normalization"] == "percentile"
    assert config["percentile_scope"] == "dataset"
    assert config["dataset_percentile_low_value"] == 100.0
    assert config["channel_tolerance_code_values"] == 2


def test_default_loss_weights_remain_l1_only_smoke_values():
    from option.Teacher import build_parser, validate_config

    args = validate_config(build_parser().parse_args([]))

    assert args.formal_training is False
    assert {name: getattr(args, name) for name in LOSS_WEIGHT_ARGUMENTS} == {
        "q_l1_weight": 1.0,
        "q_gradient_weight": 0.0,
        "q_ssim_weight": 0.0,
        "rec_l1_weight": 1.0,
        "rec_gradient_weight": 0.0,
        "rec_ssim_weight": 0.0,
        "boundary_l1_weight": 1.0,
        "boundary_gradient_weight": 0.0,
    }


def test_formal_training_and_loss_weights_share_explicit_cli_tracking():
    from option.Teacher import build_parser

    args = build_parser().parse_args([
        "--formal_training", "--q_gradient_weight", "0.7",
    ])

    assert set(args._explicit_training_objective_keys) == {
        "formal_training", "q_gradient_weight",
    }


def test_source_loss_defaults_and_formal_preset_are_owned_by_option_teacher():
    from option.Teacher import FORMAL_TRAINING_LOSS_WEIGHTS, LOSS_WEIGHT_NAMES

    assert set(FORMAL_TRAINING_LOSS_WEIGHTS) == set(LOSS_WEIGHT_NAMES)


def test_formal_training_parser_loads_positive_composite_weights():
    module = importlib.import_module("option.Teacher")

    args = module.validate_config(module.build_parser().parse_args(["--formal_training"]))

    assert args.formal_training is True
    assert {name: getattr(args, name) for name in LOSS_WEIGHT_ARGUMENTS} == FORMAL_LOSS_WEIGHTS


@pytest.mark.parametrize("name", LOSS_WEIGHT_ARGUMENTS)
def test_formal_training_rejects_each_explicit_zero_weight(name):
    from option.Teacher import build_parser, validate_config

    with pytest.raises(ValueError, match="smoke"):
        validate_config(build_parser().parse_args(["--formal_training", f"--{name}", "0"]))


@pytest.mark.parametrize("name", LOSS_WEIGHT_ARGUMENTS)
def test_loss_weights_must_always_be_non_negative(name):
    from option.Teacher import build_parser, validate_config

    with pytest.raises(ValueError, match="non-negative"):
        validate_config(build_parser().parse_args([f"--{name}", "-0.1"]))
