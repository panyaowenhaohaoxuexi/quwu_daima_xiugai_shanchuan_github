from types import SimpleNamespace

import torch

from model import FogRoutedRGBTIRDehazer
from training.source import compute_physical_mask_batch_losses


def _args():
    return SimpleNamespace(route_tau_start=1.0, route_tau_end=0.2,
                           route_temperature_anneal_steps=10, route_teacher_anneal_steps=10,
                           density_smooth_l1_beta=0.1, lambda_density=1.0, lambda_route=1.0)


def test_source_step_uses_gt_then_predicted_continuous_gate():
    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2)
    hazy, clear, tir = (torch.rand(1, 3, 32, 32) for _ in range(3))
    density, mask = torch.rand(1, 1, 32, 32), torch.ones(1, 1, 32, 32)
    initial = compute_physical_mask_batch_losses(model, (hazy, clear, tir, density, mask), _args(), global_step=0)
    late = compute_physical_mask_batch_losses(model, (hazy, clear, tir, density, mask), _args(), global_step=10)

    assert torch.equal(initial["gate"], mask)
    assert torch.equal(late["gate"], late["output"]["route_soft"])
    assert initial["losses"]["total"].requires_grad
    assert set(initial["losses"]) == {"reconstruction", "density", "route", "total", "pos_weight"}
