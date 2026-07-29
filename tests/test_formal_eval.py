import torch
from PIL import Image

from Eval import evaluate_directory
from model import FogRoutedRGBTIRDehazer


def test_eval_writes_hard_route_prediction_at_original_resolution(tmp_path):
    hazy_dir, tir_dir, output_dir = tmp_path / "hazy", tmp_path / "tir", tmp_path / "out"
    hazy_dir.mkdir(); tir_dir.mkdir()
    Image.new("RGB", (9, 7), (50, 50, 50)).save(hazy_dir / "sample.png")
    Image.new("L", (9, 7), 80).save(tir_dir / "sample.png")
    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2).eval()
    evaluate_directory(model, {"route_tau_end": 0.2, "pair_alignment_policy": "strict", "tir_normalization": "dtype_range"},
                       hazy_dir, tir_dir, output_dir, torch.device("cpu"))
    with Image.open(output_dir / "sample.png") as output:
        assert output.size == (9, 7)
