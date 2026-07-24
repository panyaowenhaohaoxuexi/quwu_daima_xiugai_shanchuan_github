import importlib
import sys


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
    ema.validate_config(ema.build_parser().parse_args([]))


def test_tir_loader_mapping_keeps_all_persisted_preprocessing_semantics():
    from option.Teacher import build_parser
    from option._formal_config import tir_normalization_config_from_args

    args = build_parser().parse_args([
        "--tir_normalization", "percentile", "--tir_percentile_scope", "dataset",
        "--tir_percentile_low", "2", "--tir_percentile_high", "98",
        "--tir_channel_tolerance_code_values", "2", "--tir_channel_tolerance_float", "0.0001",
    ])
    config = tir_normalization_config_from_args(args)
    assert config["normalization"] == "percentile"
    assert config["percentile_scope"] == "dataset"
    assert config["channel_tolerance_code_values"] == 2
