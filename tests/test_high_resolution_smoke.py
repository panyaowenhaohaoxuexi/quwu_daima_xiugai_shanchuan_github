import pytest
import torch
from PIL import Image
from types import SimpleNamespace

from model import FogRoutedRGBTIRDehazer
from training.source import OmegaSampler, compute_source_batch_losses
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


def _source_args():
    return SimpleNamespace(
        route_tau_start=1.0, route_tau_end=0.2, route_hard_start_step=1,
        counterfactual_start_step=0, route_loss_start_step=0, route_loss_warmup_steps=0,
        binary_loss_start_step=0, binary_loss_warmup_steps=0, lambda_route=1.0,
        lambda_binary=1.0, q_temperature=0.1, counterfactual_chunk_size=2,
        density_smooth_l1_beta=0.1, lambda_global=1.0, lambda_fuse=1.0,
        lambda_comp=1.0, lambda_boundary=1.0, lambda_router=1.0, lambda_density=1.0,
        rec_l1_weight=1.0, rec_gradient_weight=0.0, rec_ssim_weight=0.0,
        boundary_l1_weight=1.0, boundary_gradient_weight=0.0,
        reconstruction_ssim_window=3, reconstruction_min_valid_support=2,
    )


def test_cuda_256_complete_source_objective_executes_counterfactual_q_and_backward(monkeypatch):
    torch.cuda.reset_peak_memory_stats()
    model = FogRoutedRGBTIRDehazer(**{key: _config()[key] for key in MODEL_CONFIG_KEYS}).cuda().train()
    torch.manual_seed(19)
    hazy, clear, tir = (torch.rand(1, 3, 256, 256, device="cuda") for _ in range(3))
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 256, device="cuda"), torch.linspace(0, 1, 256, device="cuda"), indexing="ij")
    density = (0.6 * yy + 0.4 * xx).unsqueeze(0).unsqueeze(0)
    decode_calls, encoder_calls = [], {"hde": 0, "rgb": 0, "tir": 0}
    original_decode = model.decode_with_route

    def wrapped_decode(*args, **kwargs):
        decode_calls.append({
            "route_mode": kwargs.get("route_mode"),
            "route_override_value": kwargs.get("route_override_value"),
            "route_override_mask": kwargs.get("route_override_mask"),
            "memory_exclude_mask": kwargs.get("memory_exclude_mask"),
            "detached_context": not args[0]["density_map"].requires_grad,
        })
        return original_decode(*args, **kwargs)

    monkeypatch.setattr(model, "decode_with_route", wrapped_decode)
    handles = [
        model.hde.register_forward_hook(lambda *_: encoder_calls.__setitem__("hde", encoder_calls["hde"] + 1)),
        model.rgb_encoder.register_forward_hook(lambda *_: encoder_calls.__setitem__("rgb", encoder_calls["rgb"] + 1)),
        model.tir_encoder.register_forward_hook(lambda *_: encoder_calls.__setitem__("tir", encoder_calls["tir"] + 1)),
    ]
    try:
        result = compute_source_batch_losses(
            model, (hazy, clear, tir, density), _source_args(),
            OmegaSampler(regions_per_image=4, min_area=16, max_area=32, seed=7, edge_threshold=10.0),
            global_step=1, omega_generator=torch.Generator(device="cuda").manual_seed(29),
        )
        loss = result["losses"]["total"]
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()

    main = [call for call in decode_calls if call["route_override_value"] is None]
    fusion = [call for call in decode_calls if call["route_override_value"] is not None and
              torch.count_nonzero(call["route_override_value"]) == 0]
    completion = [call for call in decode_calls if call["route_override_value"] is not None and
                  torch.count_nonzero(call["route_override_value"]) > 0]
    assert len(main) >= 1 and len(fusion) >= 1 and len(completion) >= 1
    assert all(call["detached_context"] for call in fusion + completion)
    assert encoder_calls == {"hde": 1, "rgb": 1, "tir": 1}
    assert result["omega"]["omega_support"].shape[0] > 0
    assert result["q"].requires_grad is False
    assert int((result["q_valid_sum"] > 0).sum()) > 0
    assert result["route_supervision"]["valid_q_region_count"] > 0
    assert result["state"]["lambda_route"] > 0 and result["state"]["lambda_binary"] > 0
    for name in ("route", "binary", "total"):
        assert torch.isfinite(result["losses"][name]).all()
    for module in (model.hde, model.router, model.decoder_stages["h2"]):
        assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in module.parameters())
    peak = torch.cuda.max_memory_allocated()
    print(f"cuda_peak_256_complete_source_bytes={peak}")
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
