import numpy as np
import torch
from PIL import Image
from torch import nn


def _write_rgb(path, value): Image.new("RGB", (32, 32), (value, value, value)).save(path)


def test_ema_source_anchor_consumes_v2_five_item_batch(tmp_path, monkeypatch):
    import EMA
    class _ZeroClip(nn.Module):
        def forward(self, prediction, _text): return prediction.mean() * 0
    monkeypatch.setattr(EMA, "initialize_coa_clip", lambda device: (_ZeroClip().to(device), torch.zeros(1, 1, device=device)))
    monkeypatch.setattr(EMA, "require_coa_clip_cuda", lambda _device: None)
    for directory in ("clear", "ir", "hazy/1_mist", "Transmission_Map_GT/1_mist", "mask_GT/1_mist", "real/hazy", "real/tir"):
        (tmp_path / directory).mkdir(parents=True, exist_ok=True)
    _write_rgb(tmp_path / "clear" / "sample.png", 90); _write_rgb(tmp_path / "ir" / "sample.png", 40)
    _write_rgb(tmp_path / "hazy" / "1_mist" / "sample.png", 120); _write_rgb(tmp_path / "real" / "hazy" / "real.png", 100); _write_rgb(tmp_path / "real" / "tir" / "real.png", 55)
    Image.fromarray(np.full((32, 32), 50000, dtype=np.uint16), mode="I;16").save(tmp_path / "Transmission_Map_GT" / "1_mist" / "sample.png")
    Image.fromarray(np.ones((32, 32), dtype=np.uint8) * 255, mode="L").save(tmp_path / "mask_GT" / "1_mist" / "sample.png")
    from Teacher import main as source_main
    source_dir = tmp_path / "source"
    source_main(["--train_data_dir", str(tmp_path), "--validation_data_dir", str(tmp_path), "--train_size", "32", "--epochs", "1", "--iters_per_epoch", "1", "--device", "cpu", "--base_channels", "8", "--memory_max_tokens", "16", "--memory_topk", "2", "--saved_model_dir", str(source_dir), "--exp_dir", str(tmp_path / "source-exp")])
    ema_dir = tmp_path / "ema"
    EMA.main(["--source_checkpoint", str(source_dir / "source_last.pt"), "--source_anchor_data_dir", str(tmp_path), "--validation_data_dir", str(tmp_path), "--real_data_dir", str(tmp_path / "real"), "--real_tir_dir", str(tmp_path / "real" / "tir"), "--epochs", "1", "--iters_per_epoch", "1", "--device", "cpu", "--saved_model_dir", str(ema_dir), "--exp_dir", str(tmp_path / "ema-exp")])
    assert torch.load(ema_dir / "ema_last.pt", map_location="cpu")["ema_global_step"] == 1
