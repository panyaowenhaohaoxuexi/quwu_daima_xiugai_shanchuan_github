from types import SimpleNamespace

import torch

from model import FogRoutedRGBTIRDehazer
from training.source import OmegaSampler, compute_source_batch_losses


def _args():
    return SimpleNamespace(
        route_tau_start=1.0, route_tau_end=0.2, route_hard_start_step=1,
        counterfactual_start_step=0, route_loss_start_step=0, route_loss_warmup_steps=0,
        binary_loss_start_step=0, binary_loss_warmup_steps=0, lambda_route=1.0,
        lambda_binary=1.0, q_temperature=0.1, counterfactual_chunk_size=2,
        density_smooth_l1_beta=0.1, lambda_global=1.0, lambda_fuse=1.0,
        lambda_comp=1.0, lambda_boundary=1.0, lambda_router=1.0,
        lambda_density=1.0, rec_l1_weight=1.0, rec_gradient_weight=0.0,
        rec_ssim_weight=0.0, boundary_l1_weight=1.0, boundary_gradient_weight=0.0,
        reconstruction_ssim_window=3, reconstruction_min_valid_support=2,
    )


def test_source_step_produces_single_graph_loss_and_counterfactual_q():
    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2)
    batch = tuple(torch.rand(1, channels, 32, 32) for channels in (3, 3, 3, 1))
    sampler = OmegaSampler(regions_per_image=4, min_area=4, max_area=16, seed=3, edge_threshold=1.0)

    result = compute_source_batch_losses(model, batch, _args(), sampler, global_step=1)

    assert result["losses"]["total"].requires_grad
    assert result["q"].requires_grad is False
    assert result["output"]["pred_clear"].shape == batch[0].shape
    assert set(result["route_supervision"]) == {
        "sampled_omega_count", "valid_q_region_count", "valid_route_pixel_count",
    }


def test_route_supervision_counts_only_regions_with_valid_q(monkeypatch):
    import training.source as source_step

    model = FogRoutedRGBTIRDehazer(base_channels=8, memory_max_tokens=16, memory_topk=2)
    batch = tuple(torch.rand(1, channels, 32, 32) for channels in (3, 3, 3, 1))
    supports = torch.zeros(3, 1, 32, 32)
    supports[0, ..., 2:4, 2:4] = 1
    supports[1, ..., 6:8, 6:8] = 1
    supports[2, ..., 10:12, 10:12] = 1

    class _Omega:
        def sample(self, *_args, **_kwargs):
            return {
                "owner_index": torch.tensor([0, 0, 0]), "omega_support": supports,
                "omega_weight": torch.ones_like(supports), "status": ["ok"] * 3,
            }

    def fake_candidates(_model, _context, _owners, _supports, *_args):
        output = torch.zeros(3, 3, 32, 32)
        return output, output

    def fake_q(*_args, **_kwargs):
        valid = torch.zeros_like(supports)
        valid[0, ..., 2:4, 2:4] = 1
        valid[2, ..., 10:12, 10:12] = 1
        return torch.full_like(valid, 0.5), valid

    monkeypatch.setattr(source_step, "_counterfactual_predictions", fake_candidates)
    monkeypatch.setattr(source_step, "compute_q", fake_q)
    result = source_step.compute_source_batch_losses(model, batch, _args(), _Omega(), global_step=1)

    assert result["route_supervision"] == {
        "sampled_omega_count": 3,
        "valid_q_region_count": 2,
        "valid_route_pixel_count": 8,
    }
