import sys
import types

import pytest
import torch
from torch import nn


class _FakeClipModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.moves = []

    def to(self, device):
        self.moves.append(device)
        return self


class _FakeTextEncoder:
    def __init__(self, _model):
        pass

    def __call__(self, embedding_prompt, tokenized_prompts):
        return embedding_prompt + tokenized_prompts.float().sum().view(1, 1)


class _FakeClipLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, prediction, text_features):
        self.calls.append((prediction, text_features))
        return prediction.mean()


def test_coa_clip_initialization_loads_vit_rn101_and_haze_prompt(monkeypatch):
    import EMA

    loaded = []
    vit, rn101 = _FakeClipModel(), _FakeClipModel()

    def load(name, device, download_root):
        loaded.append((name, device, download_root))
        return (vit if name == "ViT-B/32" else rn101), None

    clip_module = types.SimpleNamespace(
        load=load,
        tokenize=lambda _prompt: torch.ones(1, 2, dtype=torch.long),
    )
    clip_package = types.SimpleNamespace(L_clip_from_feature=_FakeClipLoss)
    monkeypatch.setitem(sys.modules, "clip", clip_module)
    monkeypatch.setitem(sys.modules, "CLIP", clip_package)
    monkeypatch.setattr(EMA, "TextEncoder", _FakeTextEncoder, raising=False)
    monkeypatch.setattr(EMA.torch, "load", lambda _path: {"module.embedding_prompt": torch.ones(1, 1)})

    criterion, text_features = EMA.initialize_coa_clip(torch.device("cpu"))

    assert [entry[0] for entry in loaded] == ["ViT-B/32", "RN101"]
    assert all(entry[2] == "./clip_model/" for entry in loaded)
    assert vit.training is False and rn101.training is False
    assert all(not parameter.requires_grad for model in (vit, rn101) for parameter in model.parameters())
    assert isinstance(criterion, _FakeClipLoss)
    assert text_features.shape == (1, 1)


def test_real_loss_returns_coa_clip_term_for_student_clear_prediction(monkeypatch):
    from training import real_adaptation

    class _Dehazer(nn.Module):
        def __init__(self, scale):
            super().__init__()
            self.scale = nn.Parameter(torch.tensor(scale))

        def forward(self, hazy, _tir, **_kwargs):
            return {
                "pred_clear": hazy * self.scale,
                "density_map": hazy[:, :1] * 0,
                "route_soft": hazy[:, :1] * 0,
            }

    monkeypatch.setattr(real_adaptation, "stability_weights", lambda *_args: (1, 1, 1))
    monkeypatch.setattr(real_adaptation, "real_consistency_loss", lambda *args, **_kwargs: {"L_real": args[0].mean() * 0, "L_R": args[0].mean() * 0})
    args = types.SimpleNamespace(
        route_tau_end=0.2,
        ema_sigma_j=0.1,
        ema_sigma_m=0.1,
        ema_sigma_r=0.1,
        ema_stability_min_weight=0.05,
        lambda_ema_j=1.0,
        lambda_ema_m=1.0,
        lambda_ema_r=1.0,
    )
    criterion = _FakeClipLoss()
    hazy = torch.ones(1, 3, 4, 4)
    student = _Dehazer(2.0)

    result = real_adaptation.real_adaptation_loss(
        _Dehazer(1.0), student, hazy, hazy, torch.Generator().manual_seed(3), args,
        clip_criterion=criterion, text_features=torch.ones(1, 1),
    )

    assert "L_clip" in result
    assert result["L_clip"].requires_grad
    assert len(criterion.calls) == 1
    result["L_clip"].backward()
    assert student.scale.grad is not None


def test_coa_clip_requires_cuda_before_the_legacy_module_is_imported():
    import EMA

    with pytest.raises(RuntimeError, match="requires --device cuda"):
        EMA.require_coa_clip_cuda(torch.device("cpu"))


def test_adaptation_loss_applies_coa_clip_weight():
    import EMA

    args = types.SimpleNamespace(w_loss_Clip=0.5, lambda_anchor=2.0)
    result = EMA.adaptation_loss(
        {"L_real": torch.tensor(1.0), "L_clip": torch.tensor(4.0)},
        {"losses": {"total": torch.tensor(3.0)}},
        args,
    )

    assert result.item() == 9.0
