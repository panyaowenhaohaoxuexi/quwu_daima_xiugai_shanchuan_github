import pytest
import torch

from model.Teacher import FogRoutedRGBTIRDehazer
from utils.checkpoint import (CHECKPOINT_FORMAT_VERSION, build_source_checkpoint,
                              build_model_from_config, load_strict_v2_state_dict, require_checkpoint_format)


def test_checkpoint_v2_format_and_strict_state_load_errors_are_explicit():
    with pytest.raises(ValueError, match="format_version must be 2"):
        require_checkpoint_format({"format_version": 1})
    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2)
    with pytest.raises(RuntimeError, match="v2 structure-appearance Transformer"):
        load_strict_v2_state_dict(model, {}, label="Source model")
    checkpoint = build_source_checkpoint(model, torch.optim.AdamW(model.parameters()), epoch=0, global_step=0,
                                         config={"base_channels": 8})
    assert checkpoint["format_version"] == CHECKPOINT_FORMAT_VERSION
    with pytest.raises(ValueError, match="checkpoint lacks model configuration"):
        build_model_from_config({"base_channels": 8})
