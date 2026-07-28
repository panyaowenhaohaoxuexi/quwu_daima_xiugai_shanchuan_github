from argparse import Namespace
from pathlib import Path

import pytest


def _source_checkpoint_config():
    from option.Teacher import build_parser, persisted_config_from_args, validate_config

    return persisted_config_from_args(validate_config(build_parser().parse_args([])))


def test_teacher_owns_source_defaults_without_import_side_effects():
    from option.Teacher import build_parser, validate_config

    args = validate_config(build_parser().parse_args([]))

    assert (args.batch_size, args.num_workers) == (1, 0)
    assert args.train_data_dir == r"F:\1_paper_pan\1_Dehaze_Paper\2_Dataset\1_main_benchmark\1_FLIR\train"


def test_local_machine_data_and_output_defaults_target_the_configured_datasets():
    from option.EMA import build_parser as build_ema_parser, real_modal_dirs_from_args
    from option.Teacher import build_parser as build_teacher_parser

    teacher = build_teacher_parser().parse_args([])
    ema = build_ema_parser().parse_args([])

    assert teacher.train_data_dir == r"F:\1_paper_pan\1_Dehaze_Paper\2_Dataset\1_main_benchmark\1_FLIR\train"
    assert teacher.exp_dir == teacher.saved_model_dir == r"E:\Github_code_upload\Multimodal_Dehaze_code\Teacher_Train"
    assert ema.real_data_dir == r"F:\1_paper_pan\1_Dehaze_Paper\2_Dataset\1_main_benchmark\2_M3FD"
    assert ema.source_anchor_data_dir == teacher.train_data_dir
    assert ema.source_checkpoint == r"E:\Github_code_upload\Multimodal_Dehaze_code\Teacher_Train\source_last.pt"
    assert ema.exp_dir == ema.saved_model_dir == r"E:\Github_code_upload\Multimodal_Dehaze_code\Student_Train"
    assert real_modal_dirs_from_args(ema) == (
        r"F:\1_paper_pan\1_Dehaze_Paper\2_Dataset\1_main_benchmark\2_M3FD\hazy",
        r"F:\1_paper_pan\1_Dehaze_Paper\2_Dataset\1_main_benchmark\2_M3FD\ir",
    )


def test_ema_parser_exposes_only_ema_runtime_and_anchor_path_arguments():
    from option.EMA import build_parser

    args = build_parser().parse_args([
        "--real_data_dir", "real-root",
        "--source_anchor_data_dir", "anchor-root",
    ])

    assert (args.real_batch_size, args.source_anchor_batch_size, args.num_workers) == (1, 1, 0)
    assert args.real_data_dir == "real-root"
    assert args.source_anchor_data_dir == "anchor-root"
    assert args.real_tir_dir == r"F:\1_paper_pan\1_Dehaze_Paper\2_Dataset\1_main_benchmark\2_M3FD\ir"
    for forbidden in (
        "base_channels", "train_data_dir", "q_l1_weight", "omega_regions_per_image",
        "allow_ema_training_override",
    ):
        assert not hasattr(args, forbidden)


def test_ema_resolution_inherits_only_source_whitelist_and_uses_cli_ema_values():
    from option.EMA import build_parser, resolve_ema_config, validate_config

    checkpoint_config = _source_checkpoint_config()
    raw = build_parser().parse_args([
        "--real_data_dir", "real-root",
        "--source_anchor_data_dir", "anchor-root",
        "--learning_rate", "0.0003",
        "--ema_decay", "0.95",
    ])

    resolved = resolve_ema_config(raw, checkpoint_config)
    validated = validate_config(Namespace(**resolved))

    assert validated.base_channels == checkpoint_config["base_channels"]
    assert validated.q_l1_weight == checkpoint_config["q_l1_weight"]
    assert validated.real_data_dir == "real-root"
    assert validated.source_anchor_data_dir == "anchor-root"
    assert validated.start_lr == pytest.approx(0.0003)
    assert validated.ema_decay == pytest.approx(0.95)
    assert "train_data_dir" not in resolved


@pytest.mark.parametrize("raw_argv, message", [
    (["--real_data_dir", "", "--source_anchor_data_dir", "anchor"], "real_data_dir"),
    (["--real_data_dir", "real", "--source_anchor_data_dir", ""], "source_anchor_data_dir"),
])
def test_ema_validation_rejects_empty_current_data_paths(raw_argv, message):
    from option.EMA import build_parser, resolve_ema_config, validate_config

    resolved = resolve_ema_config(build_parser().parse_args(raw_argv), _source_checkpoint_config())

    with pytest.raises(ValueError, match=message):
        validate_config(Namespace(**resolved))


def test_ema_rejects_checkpoint_missing_required_source_field():
    from option.EMA import build_parser, resolve_ema_config

    checkpoint_config = _source_checkpoint_config()
    checkpoint_config.pop("memory_topk")
    raw = build_parser().parse_args([
        "--real_data_dir", "real-root", "--source_anchor_data_dir", "anchor-root",
    ])

    with pytest.raises(ValueError, match="memory_topk"):
        resolve_ema_config(raw, checkpoint_config)


def test_ema_checkpoint_config_is_whitelisted_and_excludes_startup_paths():
    from option.EMA import build_ema_checkpoint_config, build_parser, resolve_ema_config, validate_config

    checkpoint_config = _source_checkpoint_config()
    raw = build_parser().parse_args([
        "--real_data_dir", "real-root",
        "--source_anchor_data_dir", "anchor-root",
        "--source_checkpoint", "source.pt",
        "--resume_checkpoint", "resume.pt",
    ])
    args = validate_config(Namespace(**resolve_ema_config(raw, checkpoint_config)))
    persisted = build_ema_checkpoint_config(checkpoint_config, args)

    assert persisted["source_anchor_data_dir"] == "anchor-root"
    assert persisted["real_data_dir"] == "real-root"
    assert persisted["base_channels"] == checkpoint_config["base_channels"]
    for forbidden in (
        "train_data_dir", "batch_size", "source_checkpoint", "resume_checkpoint",
    ):
        assert forbidden not in persisted
    assert not any(key.startswith("_") for key in persisted)


def test_stage_modules_do_not_depend_on_the_removed_shared_option_module():
    root = Path(__file__).resolve().parents[1]
    forbidden = "_formal" + "_config"
    for path in (root / "option" / "Teacher.py", root / "option" / "EMA.py", root / "Teacher.py", root / "EMA.py"):
        assert forbidden not in path.read_text(encoding="utf-8")
