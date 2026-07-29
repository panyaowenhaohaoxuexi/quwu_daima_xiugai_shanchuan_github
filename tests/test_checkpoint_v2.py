import copy

import pytest
import torch

from model.Teacher import FogRoutedRGBTIRDehazer
from utils.checkpoint import (CHECKPOINT_FORMAT_VERSION, MODEL_CONFIG_KEYS, build_ema_checkpoint,
                              build_model_from_config, build_source_checkpoint, load_ema_checkpoint,
                              load_source_checkpoint, load_strict_v2_state_dict, require_checkpoint_format,
                              require_complete_model_config, require_integer_checkpoint_field,
                              require_state_dict_field, preflight_ema_resume_checkpoint,
                              preflight_source_initialization_checkpoint)


def _model_config():
    return {
        "base_channels": 8, "router_hidden_channels": 8, "deform_num_samples": 4,
        "deform_max_offset": 2.0, "num_structure_renderers": 2, "memory_max_tokens": 16,
        "memory_topk": 2, "memory_query_chunk_size": 64, "memory_attention_temperature": 0.07,
        "memory_reliability_epsilon": 1e-6, "memory_reliable_ratio_threshold": 0.01,
        "memory_confidence_threshold": 0.1, "decoder_num_heads": 4, "decoder_depth": 1,
        "decoder_window_size": 7, "decoder_window_chunk_size": 128, "decoder_mlp_ratio": 4.0,
        "decoder_attention_dropout": 0.0, "decoder_projection_dropout": 0.0,
        "decoder_ffn_dropout": 0.0,
    }


def test_checkpoint_v2_format_and_strict_state_load_errors_are_explicit():
    with pytest.raises(ValueError, match="format_version must be 2"):
        require_checkpoint_format({"format_version": 1})
    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2)
    with pytest.raises(RuntimeError, match="不兼容 V2 物理 mask 路由架构"):
        load_strict_v2_state_dict(model, {}, label="Source model")
    checkpoint = build_source_checkpoint(model, torch.optim.AdamW(model.parameters()), epoch=0, global_step=0,
                                         config=_model_config())
    assert checkpoint["format_version"] == CHECKPOINT_FORMAT_VERSION
    with pytest.raises(ValueError, match="checkpoint lacks model configuration"):
        build_model_from_config({"base_channels": 8})


def test_v2_preflight_rejects_monotonic_router_or_old_route_configuration():
    config = _model_config()
    config["counterfactual_start_step"] = 1
    with pytest.raises(ValueError, match="不兼容 V2 物理 mask 路由架构"):
        require_complete_model_config(config)
    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2)
    legacy_state = dict(model.state_dict())
    legacy_state["router.raw_weight_in"] = torch.ones(8)
    with pytest.raises(RuntimeError, match="不兼容 V2 物理 mask 路由架构"):
        load_strict_v2_state_dict(model, legacy_state, label="Source model")


def test_v2_checkpoint_builders_require_complete_model_config_without_runtime_fields():
    config = _model_config()
    model = build_model_from_config(config)
    optimizer = torch.optim.AdamW(model.parameters())
    source = build_source_checkpoint(model, optimizer, epoch=1, global_step=2, config=config)
    ema = build_ema_checkpoint(model, model, optimizer, epoch=1, source_global_step=2, ema_global_step=3,
                               config=config)
    assert source["format_version"] == ema["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert set(MODEL_CONFIG_KEYS) <= set(source["config"])
    assert "train_data_dir" not in source["config"]
    with pytest.raises(TypeError, match="checkpoint config must be a mapping"):
        require_complete_model_config([])
    for missing_key in ("base_channels", "decoder_depth"):
        broken = dict(config)
        broken.pop(missing_key)
        with pytest.raises(ValueError, match="checkpoint lacks model configuration"):
            build_source_checkpoint(model, optimizer, epoch=0, global_step=0, config=broken)


def test_formal_loaders_report_schema_state_and_optimizer_fields_in_validation_order():
    config = _model_config()
    model = build_model_from_config(config)
    optimizer = torch.optim.AdamW(model.parameters())
    source = build_source_checkpoint(model, optimizer, epoch=1, global_step=2, config=config)
    target = build_model_from_config(config)
    target_optimizer = torch.optim.AdamW(target.parameters())
    for mutation, error in (
        (lambda value: value.update(format_version=1), "format_version must be 2"),
        (lambda value: value.update(training_stage="ema"), "training_stage must be 'source'"),
        (lambda value: value.pop("config"), "checkpoint lacks config"),
        (lambda value: value.update(config=[]), "checkpoint config must be a mapping"),
        (lambda value: value["config"].update(decoder_depth=0.5), "decoder_depth"),
        (lambda value: value.pop("model"), "checkpoint lacks Source model state"),
        (lambda value: value.update(model=[]), "Source model state_dict must be a mapping"),
    ):
        broken = copy.deepcopy(source)
        mutation(broken)
        with pytest.raises((TypeError, ValueError), match=error):
            load_source_checkpoint(broken, target)
    missing_optimizer = dict(source)
    missing_optimizer.pop("optimizer")
    with pytest.raises(ValueError, match="checkpoint lacks Source optimizer state") as captured:
        load_source_checkpoint(missing_optimizer, target, target_optimizer)
    assert captured.value.__cause__ is None
    ema = build_ema_checkpoint(model, model, optimizer, epoch=1, source_global_step=2, ema_global_step=3,
                               config=config)
    for field, error in (("student", "EMA student state"), ("teacher", "EMA teacher state")):
        broken = dict(ema)
        broken.pop(field)
        with pytest.raises(ValueError, match=f"checkpoint lacks {error}"):
            load_ema_checkpoint(broken, target, build_model_from_config(config))


def test_checkpoint_preflight_requires_nonempty_mapping_states_and_integer_metadata():
    checkpoint = {"model": {}, "epoch": True, "global_step": "3"}
    with pytest.raises(ValueError, match="Source model state_dict must not be empty"):
        require_state_dict_field(checkpoint, "model", label="Source model")
    checkpoint["model"] = []
    with pytest.raises(TypeError, match="Source model state_dict must be a mapping"):
        require_state_dict_field(checkpoint, "model", label="Source model")
    with pytest.raises(TypeError, match="epoch must be an integer"):
        require_integer_checkpoint_field(checkpoint, "epoch")
    checkpoint["epoch"] = 0.5
    with pytest.raises(TypeError, match="epoch must be an integer"):
        require_integer_checkpoint_field(checkpoint, "epoch")
    assert require_integer_checkpoint_field(checkpoint, "global_step") == 3


def test_source_initialization_and_ema_resume_preflight_require_their_exact_fields():
    config = _model_config()
    model = build_model_from_config(config)
    optimizer = torch.optim.AdamW(model.parameters())
    source = build_source_checkpoint(model, optimizer, epoch=1, global_step=2, config=config)
    assert preflight_source_initialization_checkpoint(source)["metadata"]["global_step"] == 2
    missing_global_step = dict(source)
    missing_global_step.pop("global_step")
    with pytest.raises(ValueError, match="checkpoint lacks global_step"):
        preflight_source_initialization_checkpoint(missing_global_step)

    ema = build_ema_checkpoint(model, model, optimizer, epoch=1, source_global_step=2, ema_global_step=3,
                               config=config)
    assert preflight_ema_resume_checkpoint(ema)["metadata"]["ema_global_step"] == 3
    for field in ("student", "teacher", "optimizer", "epoch", "source_global_step", "ema_global_step"):
        broken = dict(ema)
        broken.pop(field)
        with pytest.raises(ValueError, match=field):
            preflight_ema_resume_checkpoint(broken)


@pytest.mark.parametrize(
    ("entry_module", "entry_args", "checkpoint_fields", "expected"),
    (
        ("Teacher", ("--resume_checkpoint",), {"optimizer": {}, "epoch": 0, "global_step": 0},
         "checkpoint lacks Source model state"),
        ("EMA", ("--source_checkpoint",), {"global_step": 0},
         "checkpoint lacks Source model state"),
        ("Eval", ("--checkpoint",), {}, "checkpoint lacks Source model state"),
    ),
)
def test_formal_entrypoints_preflight_broken_state_before_any_construction(
        tmp_path, monkeypatch, entry_module, entry_args, checkpoint_fields, expected):
    """A malformed formal checkpoint must fail before constructors or output directories run."""
    import importlib

    checkpoint = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "training_stage": "source",
        "config": _model_config(),
        **checkpoint_fields,
    }
    checkpoint_path = tmp_path / f"{entry_module}.pt"
    torch.save(checkpoint, checkpoint_path)
    module = importlib.import_module(entry_module)
    calls = []

    def forbidden(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("construction side effect occurred before checkpoint preflight")

    for name in ("build_model_from_config", "prepare_experiment_dirs", "SynthMultiModalDataset", "DataLoader"):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, forbidden)
    argv = [*entry_args, str(checkpoint_path)]
    if entry_module == "Teacher":
        argv += ["--train_data_dir", str(tmp_path)]
    elif entry_module == "EMA":
        argv += ["--real_data_dir", str(tmp_path), "--source_anchor_data_dir", str(tmp_path)]
    else:
        argv += ["--hazy_dir", str(tmp_path), "--tir_dir", str(tmp_path), "--output_dir", str(tmp_path / "out")]
    with pytest.raises(ValueError, match=expected):
        module.main(argv)
    assert not calls
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    ("entry_module", "checkpoint", "entry_args", "expected"),
    (
        ("Teacher", {"training_stage": "source", "model": {"weight": torch.ones(1)}, "optimizer": [],
                     "epoch": 0, "global_step": 0}, ("--resume_checkpoint",), "Source optimizer state must be a mapping"),
        ("EMA", {"training_stage": "source", "model": {"weight": torch.ones(1)}, "global_step": True},
         ("--source_checkpoint",), "global_step must be an integer"),
        ("EMA", {"training_stage": "ema", "student": {"weight": torch.ones(1)},
                 "teacher": {"weight": torch.ones(1)}, "optimizer": {}, "epoch": 0,
                 "source_global_step": True, "ema_global_step": 0},
         ("--resume_checkpoint",), "source_global_step must be an integer"),
    ),
)
def test_entrypoint_preflight_metadata_and_optimizer_fail_before_side_effects(
        tmp_path, monkeypatch, entry_module, checkpoint, entry_args, expected):
    import importlib

    checkpoint = {"format_version": CHECKPOINT_FORMAT_VERSION, "config": _model_config(), **checkpoint}
    checkpoint_path = tmp_path / f"{entry_module}-metadata.pt"
    torch.save(checkpoint, checkpoint_path)
    module = importlib.import_module(entry_module)
    calls = []

    def forbidden(*_args, **_kwargs):
        calls.append(True)
        raise AssertionError("construction or RNG side effect occurred before checkpoint preflight")

    for name in ("_set_seed", "build_model_from_config", "prepare_experiment_dirs", "save_config",
                 "SynthMultiModalDataset", "RealMultiModalDataset", "DataLoader", "AdamW"):
        if hasattr(module, name):
            monkeypatch.setattr(module, name, forbidden)
    argv = [*entry_args, str(checkpoint_path)]
    if entry_module == "Teacher":
        argv += ["--train_data_dir", str(tmp_path)]
    else:
        argv += ["--real_data_dir", str(tmp_path), "--source_anchor_data_dir", str(tmp_path)]
    with pytest.raises((TypeError, ValueError), match=expected):
        module.main(argv)
    assert not calls
