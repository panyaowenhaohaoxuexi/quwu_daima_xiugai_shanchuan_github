import copy

import pytest
import torch

from model.Teacher import FogRoutedRGBTIRDehazer
from utils.checkpoint import (CHECKPOINT_FORMAT_VERSION, MODEL_CONFIG_KEYS, build_ema_checkpoint,
                              build_model_from_config, build_source_checkpoint, load_ema_checkpoint,
                              load_source_checkpoint, load_strict_v2_state_dict, require_checkpoint_format,
                              require_complete_model_config)


def _model_config():
    return {
        "base_channels": 8, "router_hidden_channels": 8, "deform_num_samples": 4,
        "deform_max_offset": 2.0, "num_structure_renderers": 2, "memory_max_tokens": 16,
        "memory_topk": 2, "memory_query_chunk_size": 64, "memory_attention_temperature": 0.07,
        "memory_reliability_epsilon": 1e-6, "memory_reliable_ratio_threshold": 0.01,
        "memory_confidence_threshold": 0.1, "memory_exclusion_extra_margin": 0,
        "boundary_width": 1, "decoder_num_heads": 4, "decoder_depth": 1,
        "decoder_window_size": 7, "decoder_window_chunk_size": 128, "decoder_mlp_ratio": 4.0,
        "decoder_attention_dropout": 0.0, "decoder_projection_dropout": 0.0,
        "decoder_ffn_dropout": 0.0,
    }


def test_checkpoint_v2_format_and_strict_state_load_errors_are_explicit():
    with pytest.raises(ValueError, match="format_version must be 2"):
        require_checkpoint_format({"format_version": 1})
    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2)
    with pytest.raises(RuntimeError, match="v2 structure-appearance Transformer"):
        load_strict_v2_state_dict(model, {}, label="Source model")
    checkpoint = build_source_checkpoint(model, torch.optim.AdamW(model.parameters()), epoch=0, global_step=0,
                                         config=_model_config())
    assert checkpoint["format_version"] == CHECKPOINT_FORMAT_VERSION
    with pytest.raises(ValueError, match="checkpoint lacks model configuration"):
        build_model_from_config({"base_channels": 8})


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
        (lambda value: value.update(model=[]), "Source model state_dict must be a dictionary"),
    ):
        broken = copy.deepcopy(source)
        mutation(broken)
        with pytest.raises((TypeError, ValueError), match=error):
            load_source_checkpoint(broken, target)
    missing_optimizer = dict(source)
    missing_optimizer.pop("optimizer")
    with pytest.raises(RuntimeError, match="Source optimizer state is incompatible") as captured:
        load_source_checkpoint(missing_optimizer, target, target_optimizer)
    assert captured.value.__cause__ is not None
    ema = build_ema_checkpoint(model, model, optimizer, epoch=1, source_global_step=2, ema_global_step=3,
                               config=config)
    for field, error in (("student", "EMA student state"), ("teacher", "EMA teacher state")):
        broken = dict(ema)
        broken.pop(field)
        with pytest.raises(ValueError, match=f"checkpoint lacks {error}"):
            load_ema_checkpoint(broken, target, build_model_from_config(config))
