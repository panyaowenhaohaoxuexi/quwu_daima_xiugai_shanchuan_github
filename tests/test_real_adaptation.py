from types import SimpleNamespace

import torch
from torch import nn

from training.real_adaptation import real_adaptation_loss


class _ToyDehazer(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(float(scale)), requires_grad=False)

    def forward(self, hazy, tir, route_temperature=1.0, route_mode="hard"):
        base = hazy[:, :1]
        return {
            "pred_clear": (self.scale * hazy).clamp(0.0, 1.0),
            "density_map": (self.scale * base).clamp(0.0, 1.0),
            "route_soft": torch.sigmoid(self.scale * base - 0.25),
        }


def _args():
    return SimpleNamespace(
        route_tau_end=0.2,
        ema_sigma_j=0.1,
        ema_sigma_m=0.1,
        ema_sigma_r=0.1,
        ema_stability_min_weight=0.05,
        lambda_ema_j=1.0,
        lambda_ema_m=1.0,
        lambda_ema_r=2.0,
    )


def test_route_multiplier_zero_excludes_only_route_consistency():
    hazy = torch.rand(1, 3, 8, 8)
    tir = torch.rand(1, 3, 8, 8)
    losses = real_adaptation_loss(
        _ToyDehazer(0.7), _ToyDehazer(0.4), hazy, tir,
        torch.Generator().manual_seed(7), _args(), route_multiplier=0.0,
    )

    assert losses["L_R"].item() == 0.0
    assert losses["L_J"].item() > 0.0
    assert losses["L_M"].item() > 0.0
    assert torch.allclose(losses["L_real"], losses["L_J"] + losses["L_M"])


def test_route_multiplier_scales_only_route_consistency():
    hazy = torch.rand(1, 3, 8, 8)
    tir = torch.rand(1, 3, 8, 8)
    low = real_adaptation_loss(
        _ToyDehazer(0.7), _ToyDehazer(0.4), hazy, tir,
        torch.Generator().manual_seed(11), _args(), route_multiplier=0.25,
    )
    high = real_adaptation_loss(
        _ToyDehazer(0.7), _ToyDehazer(0.4), hazy, tir,
        torch.Generator().manual_seed(11), _args(), route_multiplier=1.0,
    )

    assert torch.allclose(high["L_J"], low["L_J"])
    assert torch.allclose(high["L_M"], low["L_M"])
    assert torch.allclose(high["L_R"], low["L_R"] * 4.0)
