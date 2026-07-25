import importlib
import importlib.util
import sys


def test_model_package_exports_only_formal_model_and_router():
    for name in list(sys.modules):
        if name == "model" or name.startswith("model."):
            del sys.modules[name]

    package = importlib.import_module("model")
    teacher_module = importlib.import_module("model.Teacher")

    assert package.__all__ == ["FogRoutedRGBTIRDehazer", "MonotonicFogRouter"]
    assert "model.Teacher" in sys.modules
    assert package.FogRoutedRGBTIRDehazer is teacher_module.FogRoutedRGBTIRDehazer


def test_legacy_model_kd_and_loss_modules_are_removed():
    removed_modules = (
        "model.fog_routed_dehazer",
        "model.cmdn",
        "model.dsfe",
        "model.gumbel_sigmoid",
        "model.teacher_color",
        "model.teacher_fusion",
        "model.teacher_semantic",
        "model.vifnet_basic_modules",
        "model.diagnose_loss",
        "model.Student",
        "model.Student_x",
        "KD",
        "option.KD",
        "loss.Feature_alignment",
        "loss.cr",
        "loss.SSIM",
        "loss.teacher_region_loss",
    )
    for module_name in removed_modules:
        assert importlib.util.find_spec(module_name) is None, module_name


def test_loss_package_keeps_formal_fog_routed_losses_importable():
    import loss
    from loss.fog_routed_source_loss import (
        masked_gradient_error,
        masked_local_ssim_error,
        masked_smooth_l1,
    )

    assert loss is not None
    assert all(callable(loss_fn) for loss_fn in (
        masked_gradient_error,
        masked_local_ssim_error,
        masked_smooth_l1,
    ))
