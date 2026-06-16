from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_real_probe_visualization_has_dedicated_six_column_overview():
    teacher_py = (ROOT / "Teacher.py").read_text(encoding="utf-8")

    assert "REAL_PROBE_COLUMNS" in teacher_py
    assert "def save_real_probe_overview" in teacher_py
    assert '["Hazy", "IR", "Pred", "Density_pred", "Mask_prob", "Binary_mask"]' in teacher_py.replace("\n", "")
    assert "{stem}_overview.png" not in teacher_py


def test_real_probe_visualization_does_not_use_gt_fields_and_restores_size():
    teacher_py = (ROOT / "Teacher.py").read_text(encoding="utf-8")
    start = teacher_py.index("def run_real_world_visualization")
    end = teacher_py.index("# --- [新增结束] ---", start)
    real_vis_body = teacher_py[start:end]

    for forbidden in ("clear_vis", "density_gt", "mask_gt", "Transmission_Map_GT", "IR_Completion_Mask_GT"):
        assert forbidden not in real_vis_body

    assert 'out = model(haze_vis_resized, haze_ir_resized, return_dict=True)' in real_vis_body
    assert 'out["pred_clear"]' in real_vis_body
    assert 'out["density_map"]' in real_vis_body
    assert 'out["mask_prob"]' in real_vis_body
    assert 'out["binary_mask"]' in real_vis_body
    assert "size=(h, w)" in real_vis_body
    assert 'mode="nearest"' in real_vis_body


def test_train_batch_region_visualization_is_default_off_and_gated():
    option_teacher_py = (ROOT / "option" / "Teacher.py").read_text(encoding="utf-8")
    teacher_py = (ROOT / "Teacher.py").read_text(encoding="utf-8")

    assert "--save_train_batch_region_vis" in option_teacher_py
    assert "default=False" in option_teacher_py
    assert 'getattr(opt, "save_train_batch_region_vis", False)' in teacher_py


def test_real_probe_visualization_trigger_is_independent_from_real_infer_flag():
    teacher_py = (ROOT / "Teacher.py").read_text(encoding="utf-8")
    eval_block = teacher_py[teacher_py.index("# 执行评估"):]

    real_test_call = eval_block.index("run_real_world_test(")
    real_vis_call = eval_block.index("run_real_world_visualization(")
    infer_guard = eval_block.rfind('getattr(opt, "run_real_infer_in_teacher", False)', 0, real_test_call)
    specific_guard = eval_block.rfind("opt.real_test_specific_hazy_dir and opt.real_test_specific_ir_dir", 0, real_vis_call)

    assert infer_guard != -1
    assert specific_guard != -1
    assert specific_guard > infer_guard
