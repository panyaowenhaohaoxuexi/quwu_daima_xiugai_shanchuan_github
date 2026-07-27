import pytest
import torch
from PIL import Image

from model import FogRoutedRGBTIRDehazer
from utils.checkpoint import CHECKPOINT_FORMAT_VERSION, MODEL_CONFIG_KEYS


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable for high-resolution smoke")


def _config():
    values = {
        "base_channels": 8, "router_hidden_channels": 8, "deform_num_samples": 4,
        "deform_max_offset": 2.0, "num_structure_renderers": 2, "memory_max_tokens": 16,
        "memory_topk": 2, "memory_query_chunk_size": 1024, "memory_attention_temperature": 0.07,
        "memory_reliability_epsilon": 1e-6, "memory_reliable_ratio_threshold": 0.01,
        "memory_confidence_threshold": 0.1, "memory_exclusion_extra_margin": 0, "boundary_width": 1,
        "decoder_num_heads": 4, "decoder_depth": 1, "decoder_window_size": 7,
        "decoder_window_chunk_size": 128, "decoder_mlp_ratio": 4.0,
        "decoder_attention_dropout": 0.0, "decoder_projection_dropout": 0.0, "decoder_ffn_dropout": 0.0,
        "route_tau_end": 0.2, "tir_normalization": "dtype_range", "tir_percentile_low": 1.0,
        "tir_percentile_high": 99.0, "tir_percentile_scope": "per_image",
        "tir_channel_tolerance_code_values": 1, "tir_channel_tolerance_float": 1e-5,
    }
    assert set(MODEL_CONFIG_KEYS) <= set(values)
    return values


def test_cuda_256_source_style_forward_backward_is_finite():
    torch.cuda.reset_peak_memory_stats()
    model = FogRoutedRGBTIRDehazer(**{key: _config()[key] for key in MODEL_CONFIG_KEYS}).cuda().train()
    hazy, clear, tir = (torch.rand(1, 3, 256, 256, device="cuda") for _ in range(3))
    output = model(hazy, tir, route_mode="soft")
    loss = (output["pred_clear"] - clear).abs().mean() + output["density_map"].mean() + output["route_soft"].mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    peak = torch.cuda.max_memory_allocated()
    print(f"cuda_peak_256_source_bytes={peak}")
    assert peak > 0


def test_cuda_1024x768_eval_save_aux_preserves_original_resolution(tmp_path):
    from Eval import main

    config = _config()
    model = FogRoutedRGBTIRDehazer(**{key: config[key] for key in MODEL_CONFIG_KEYS})
    checkpoint = tmp_path / "source.pt"
    torch.save({"format_version": CHECKPOINT_FORMAT_VERSION, "training_stage": "source", "config": config,
                "model": model.state_dict()}, checkpoint)
    hazy_dir, tir_dir, out_dir = tmp_path / "hazy", tmp_path / "tir", tmp_path / "out"
    hazy_dir.mkdir(); tir_dir.mkdir()
    Image.new("RGB", (1024, 768), (80, 80, 80)).save(hazy_dir / "sample.png")
    Image.new("L", (1024, 768), 100).save(tir_dir / "sample.png")
    torch.cuda.reset_peak_memory_stats()
    main(["--checkpoint", str(checkpoint), "--hazy_dir", str(hazy_dir), "--tir_dir", str(tir_dir),
          "--output_dir", str(out_dir), "--device", "cuda", "--save_aux"])
    assert Image.open(out_dir / "sample.png").size == (1024, 768)
    assert (out_dir / "aux" / "sample_density_map.png").is_file()
    peak = torch.cuda.max_memory_allocated()
    print(f"cuda_peak_1024_eval_bytes={peak}")
    assert peak > 0
