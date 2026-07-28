from pathlib import Path

import torch

from model.Teacher import FogRoutedRGBTIRDehazer, PyramidEncoder


def test_rgb_uses_coa_res2net_pretraining_while_tir_remains_independent():
    model = FogRoutedRGBTIRDehazer(
        base_channels=8,
        memory_max_tokens=16,
        memory_topk=2,
    ).eval()

    expected_path = Path(__file__).resolve().parents[1] / "model" / "imagenet_model" / "res2net101_v1b_26w_4s-0812c246.pth"
    assert model.rgb_encoder.pretrained_path == expected_path
    assert isinstance(model.tir_encoder, PyramidEncoder)
    assert not hasattr(model.tir_encoder, "pretrained_path")
    pretrained = torch.load(expected_path, map_location="cpu")
    assert torch.equal(model.rgb_encoder.encoder.conv1[0].weight, pretrained["conv1.0.weight"])

    with torch.no_grad():
        features = model.rgb_encoder(torch.randn(1, 3, 32, 32))

    assert {name: value.shape[1] for name, value in features.items()} == {
        "h2": 8,
        "h4": 16,
        "h8": 24,
        "h16": 32,
    }
    assert {name: value.shape[-2:] for name, value in features.items()} == {
        "h2": (16, 16),
        "h4": (8, 8),
        "h8": (4, 4),
        "h16": (2, 2),
    }
