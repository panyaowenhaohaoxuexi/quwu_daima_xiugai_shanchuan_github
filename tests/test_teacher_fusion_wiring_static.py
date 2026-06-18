from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_teacher_options_define_only_the_three_new_fusion_controls():
    text = (ROOT / "option" / "Teacher.py").read_text(encoding="utf-8")

    assert "parser.add_argument('--fusion_temperature', default=0.07, type=float" in text
    assert "parser.add_argument('--verify_threshold', default=0.2, type=float" in text
    assert "parser.add_argument('--verify_temperature', default=0.1, type=float" in text
    for existing in (
        "w_loss_comp_perc",
        "align_mode",
        "align_temperature",
        "infonce_fp_threshold",
        "infonce_max_samples",
        "infonce_fp_warmup_steps",
    ):
        assert text.count(f"parser.add_argument('--{existing}'") == 1


def test_training_entry_passes_all_fusion_controls_to_teacher():
    text = (ROOT / "Teacher.py").read_text(encoding="utf-8")

    assert "fusion_temperature=opt.fusion_temperature" in text
    assert "verify_threshold=opt.verify_threshold" in text
    assert "verify_temperature=opt.verify_temperature" in text
