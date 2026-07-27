import torch

from training.source import OmegaSampler


def test_omega_sampler_returns_detached_nonoverlapping_support_and_weights():
    density = torch.linspace(0, 1, 32 * 32, requires_grad=True).reshape(1, 1, 32, 32)
    edges = torch.zeros_like(density, requires_grad=True)
    sampler = OmegaSampler(regions_per_image=4, min_area=8, max_area=32, seed=7)

    result = sampler.sample(density, edges)

    assert result["omega_support"].requires_grad is False
    assert result["omega_weight"].requires_grad is False
    assert result["owner_index"].numel() == result["status"][0]["actual_count"]
    support = result["omega_support"]
    assert torch.all(support.sum(dim=0) <= 1)
    assert torch.all((result["omega_weight"] >= 0) & (result["omega_weight"] <= 1))


def test_flat_density_and_tir_sampler_is_deterministic_and_reports_status():
    density = torch.full((1, 1, 32, 32), 0.5)
    edges = torch.zeros_like(density)
    sampler = OmegaSampler(regions_per_image=4, min_area=4, max_area=16, seed=11)

    first = sampler.sample(density, edges)
    second = sampler.sample(density, edges)

    assert torch.equal(first["omega_support"], second["omega_support"])
    status = first["status"][0]
    assert status["requested_count"] == 4
    assert "fallback_reason" in status


def test_feasible_sample_covers_density_quantiles_without_overlap():
    density = torch.linspace(0, 1, 48 * 48).reshape(1, 1, 48, 48)
    sampler = OmegaSampler(regions_per_image=6, min_area=9, max_area=36, seed=3)
    result = sampler.sample(density, torch.zeros_like(density))

    assert result["status"][0]["actual_count"] == 6
    assert {"low", "middle", "high"}.issubset(set(result["quantile_class"]))
    assert torch.all(result["omega_support"].sum(dim=0) <= 1)


def test_strong_tir_edge_blocks_connected_region_growth():
    density = torch.full((1, 1, 32, 32), 0.5)
    edge = torch.zeros_like(density)
    edge[..., :, 16] = 1.0
    sampler = OmegaSampler(regions_per_image=4, min_area=9, max_area=25, seed=19, edge_threshold=0.5)
    result = sampler.sample(density, edge)

    for support in result["omega_support"]:
        occupied_columns = torch.where(support[0].any(dim=0))[0]
        assert not ((occupied_columns < 16).any() and (occupied_columns > 16).any())
