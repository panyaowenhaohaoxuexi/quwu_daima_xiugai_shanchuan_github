from argparse import Namespace


def _source_checkpoint_config():
    from option.Teacher import build_parser, persisted_config_from_args, validate_config

    args = validate_config(build_parser().parse_args([]))
    return persisted_config_from_args(args)


def test_teacher_owns_source_defaults_without_import_side_effects():
    from option.Teacher import build_parser, validate_config

    args = validate_config(build_parser().parse_args([]))

    assert args.batch_size == 1
    assert args.num_workers == 0
    assert args.train_data_dir == ""
    assert args.q_l1_weight == 1.0
    assert args.q_gradient_weight == 0.0
    assert args.formal_training is False


def test_ema_parser_exposes_only_ema_runtime_and_anchor_path_arguments():
    from option.EMA import build_parser

    args = build_parser().parse_args([
        "--real_data_dir", "real-root",
        "--source_anchor_data_dir", "anchor-root",
    ])

    assert args.real_data_dir == "real-root"
    assert args.source_anchor_data_dir == "anchor-root"
    assert not hasattr(args, "base_channels")
    assert not hasattr(args, "train_data_dir")
    assert not hasattr(args, "q_l1_weight")


def test_ema_resolution_inherits_source_semantics_and_maps_legacy_anchor_path():
    from option.EMA import build_parser, resolve_ema_config, validate_config

    checkpoint_config = _source_checkpoint_config()
    checkpoint_config["train_data_dir"] = "legacy-source-root"
    raw = build_parser().parse_args(["--real_data_dir", "real-root"])

    resolved, diff = resolve_ema_config(
        raw, checkpoint_config, allow_training_override=False,
    )
    validated = validate_config(Namespace(**resolved))

    assert validated.base_channels == checkpoint_config["base_channels"]
    assert validated.q_l1_weight == checkpoint_config["q_l1_weight"]
    assert validated.source_anchor_data_dir == "legacy-source-root"
    assert validated.real_data_dir == "real-root"
    assert diff == {}


def test_ema_checkpoint_config_uses_new_anchor_field_and_cli_anchor_takes_precedence():
    from option.EMA import (
        build_ema_checkpoint_config,
        build_parser,
        resolve_ema_config,
        validate_config,
    )

    checkpoint_config = _source_checkpoint_config()
    checkpoint_config["train_data_dir"] = "legacy-source-root"
    raw = build_parser().parse_args([
        "--real_data_dir", "real-root",
        "--source_anchor_data_dir", "new-anchor-root",
    ])

    resolved, _ = resolve_ema_config(raw, checkpoint_config, allow_training_override=False)
    args = validate_config(Namespace(**resolved))
    persisted = build_ema_checkpoint_config(checkpoint_config, args)

    assert persisted["source_anchor_data_dir"] == "new-anchor-root"
    assert "train_data_dir" not in persisted
    assert persisted["base_channels"] == checkpoint_config["base_channels"]


def test_stage_modules_do_not_depend_on_the_removed_shared_option_module():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    forbidden = "_formal" + "_config"
    for path in (root / "option" / "Teacher.py", root / "option" / "EMA.py", root / "Teacher.py", root / "EMA.py"):
        assert forbidden not in path.read_text(encoding="utf-8")
