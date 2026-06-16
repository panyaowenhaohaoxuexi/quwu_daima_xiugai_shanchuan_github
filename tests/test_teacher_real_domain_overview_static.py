from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def _function_body(source, name):
    start = source.index(f"def {name}")
    next_def = source.find("\ndef ", start + 1)
    end = len(source) if next_def == -1 else next_def
    return source[start:end]


def test_specific_probe_options_and_visualization_entrypoint_are_removed():
    teacher_py = _read("Teacher.py")
    option_teacher_py = _read("option/Teacher.py")

    for forbidden in (
        "real_test_" + "specific_hazy_dir",
        "real_test_" + "specific_ir_dir",
        "run_real_world_" + "visualization",
        "[Real" + "Vis]",
        "[Real" + "VisDebug]",
    ):
        assert forbidden not in teacher_py

    assert "real_test_" + "specific_hazy_dir" not in option_teacher_py
    assert "real_test_" + "specific_ir_dir" not in option_teacher_py


def test_real_domain_overview_uses_real_test_output_dir_not_saved_data_legacy_path():
    teacher_py = _read("Teacher.py")
    option_teacher_py = _read("option/Teacher.py")
    run_real_test_body = _function_body(teacher_py, "run_real_world_test")

    for source in (teacher_py, option_teacher_py):
        assert 'os.path.join(opt.saved_data_dir, "' + "real_vis" + '"' not in source
        assert "saved_data_dir/" + "real_vis" not in source
        assert '"' + "real_vis" + '"' not in source

    assert "opt.real_test_output_dir" in run_real_test_body
    assert "save_real_probe_overview(samples, save_path)" in run_real_test_body
    assert 'out = model(haze_vis_resized, haze_ir_resized, return_dict=True)' in run_real_test_body
    assert "size=(h, w)" in run_real_test_body


def test_real_domain_overview_columns_are_prediction_only():
    teacher_py = _read("Teacher.py")

    assert "REAL_PROBE_COLUMNS" in teacher_py
    assert '["Hazy", "IR", "Pred", "Density_pred", "Mask_prob", "Binary_mask"]' in teacher_py.replace("\n", "")

    run_real_test_body = _function_body(teacher_py, "run_real_world_test")
    for forbidden in ("Clear", "Density_gt", "Mask_gt", "clear_vis", "density_gt", "mask_gt"):
        assert forbidden not in run_real_test_body


def test_train_real_domain_trigger_is_only_run_real_infer_in_teacher():
    teacher_py = _read("Teacher.py")
    eval_block = teacher_py[teacher_py.index("# 执行评估"):]

    assert 'getattr(opt, "run_real_infer_in_teacher", False)' in eval_block
    assert "run_real_world_test(" in eval_block
    assert "opt.real_test_hazy_path" in eval_block
    assert "opt.real_test_ir_path" in eval_block
    assert "run_real_world_" + "visualization" not in eval_block
    assert "real_test_" + "specific" not in eval_block


def test_train_batch_region_visualization_remains_default_off_and_gated():
    option_teacher_py = _read("option/Teacher.py")
    teacher_py = _read("Teacher.py")

    assert "--save_train_batch_region_vis" in option_teacher_py
    assert "default=False" in option_teacher_py
    assert 'getattr(opt, "save_train_batch_region_vis", False)' in teacher_py
    assert "save_teacher_region_visualization(" in teacher_py
