from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(path):
    return (ROOT / path).read_text(encoding="utf-8")


def test_eval_ema_uses_internal_mask_outputs_only():
    source = _read("Eval_EMA.py")

    for forbidden in (
        "INPUT_FOLDER_MASK",
        "MIN_DEHAZE_STRENGTH",
        "transform_mask",
        "transform_to_tensor_only",
        "mask_image_path",
        "alpha_final",
        "use_mask_if_available",
        "haze_mask_tensor",
        "haze_mask_resized",
        "1.0 - alpha",
        "haze_vis_original",
        "original_tensor",
    ):
        assert forbidden not in source

    assert "def dehaze(model, vis_image_path, ir_image_path, folder):" in source
    assert "return_dict=True" in source
    assert 'out["pred_clear"]' in source
    assert 'out["density_map"]' in source
    assert 'out["mask_prob"]' in source
    assert 'out["binary_mask"]' in source


def test_eval_ema_has_no_training_import_or_external_mask_override():
    source = _read("Eval_EMA.py")

    assert "from Teacher import" not in source
    assert "import Teacher" not in source
    assert "haze_mask=" not in source
    assert "debug_force_mask=" not in source
    assert "model(..., haze_mask" not in source


def test_eval_ema_saves_restored_pred_and_internal_overview():
    source = _read("Eval_EMA.py")

    assert "SAVE_INTERNAL_OVERVIEW" in source
    assert "internal_mask_vis" in source
    assert "model.eval()" in source
    assert "torch.no_grad()" in source
    assert "pred_to_save = pred_clear_restored.squeeze(0).clamp(0, 1)" in source
    assert "torchvision.utils.save_image(pred_to_save, save_path)" in source


def test_teacher_comments_mark_cmdn_legacy_and_active_gumbel_path():
    source = _read("model/Teacher.py")
    lowered = source.lower()

    assert "self.cmdn = None" in source
    assert "HDE" in source
    assert "GumbelSigmoidBinarizer" in source
    assert "legacy" in lowered or "disabled" in lowered
