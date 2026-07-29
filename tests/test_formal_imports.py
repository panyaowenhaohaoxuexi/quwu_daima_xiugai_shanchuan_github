import importlib
import importlib.util
import sys
from pathlib import Path


def test_model_package_exports_only_formal_model_and_router():
    for name in list(sys.modules):
        if name == "model" or name.startswith("model."):
            del sys.modules[name]

    package = importlib.import_module("model")
    teacher_module = importlib.import_module("model.Teacher")

    assert package.__all__ == ["FogRoutedRGBTIRDehazer", "FeatureGuidedRouter"]
    assert importlib.util.find_spec("model.fog_routed_dehazer") is None
    assert package.FogRoutedRGBTIRDehazer is teacher_module.FogRoutedRGBTIRDehazer


def test_legacy_model_kd_and_loss_modules_are_removed():
    removed_modules = (
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
    from loss.common import (
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


def test_data_package_exports_only_formal_rgb_tir_interfaces():
    import data
    import data.data_loader as loader

    assert all(hasattr(data, name) for name in (
        "SynthMultiModalDataset", "RealMultiModalDataset", "collate_synth", "collate_real",
        "load_tir_as_float_tensor", "load_scalar_map_as_float_tensor", "convert_density_semantics",
    ))
    assert all(not hasattr(data, name) for name in (
        "RESIDE_Dataset", "TestDataset", "CLIP_loader", "RESIDE_Dataset_2",
        "MultiModalHazeDataset", "MultiModalCLIPLoader", "StatefulRandomSampler",
    ))
    assert all(not hasattr(loader, name) for name in (
        "RESIDE_Dataset", "TestDataset", "CLIP_loader", "RESIDE_Dataset_2",
        "MultiModalHazeDataset", "MultiModalCLIPLoader",
    ))


def test_replaced_training_loss_and_sampler_modules_are_absent():
    removed = (
        "training.source_step", "training.source_counterfactual", "training.omega_sampler",
        "training.omega_state", "training.ema_core", "training.ema_step", "training.paired_geometry",
        "training.checkpointing", "training.resume_config", "training.schedules", "training.source_objective",
        "training.step_control", "training.step_transaction", "data.stateful_sampler",
        "loss.synthetic", "loss.real", "loss.fog_routed_source_loss",
    )
    assert all(importlib.util.find_spec(module) is None for module in removed)


def test_teacher_model_imports_and_strict_state_dict_are_self_consistent():
    import torch
    from model import FogRoutedRGBTIRDehazer as package_model
    from model.Teacher import FogRoutedRGBTIRDehazer as module_model

    reference = module_model(base_channels=8, memory_max_tokens=16, memory_topk=2)
    restored = package_model(base_channels=8, memory_max_tokens=16, memory_topk=2)
    assert set(reference.state_dict()) == set(restored.state_dict())
    assert {key: value.shape for key, value in reference.state_dict().items()} == {
        key: value.shape for key, value in restored.state_dict().items()
    }
    restored.load_state_dict(reference.state_dict(), strict=True)
    rgb, tir = torch.rand(1, 3, 32, 32), torch.rand(1, 3, 32, 32)
    expected, actual = reference(rgb, tir, route_mode="hard"), restored(rgb, tir, route_mode="hard")
    for key in expected:
        torch.testing.assert_close(expected[key], actual[key], rtol=0, atol=0)


def test_readme_distinguishes_training_entrypoints_model_definition_and_formal_command():
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "`Teacher.py`" in readme and "Source" in readme
    assert "`model/Teacher.py`" in readme and "FogRoutedRGBTIRDehazer" in readme
    assert "`EMA.py`" in readme and "`Eval.py`" in readme
    assert "not a third training stage" in readme
    assert "model/fog_routed_dehazer.py" in readme and "does not exist" in readme
