import pytest
import torch

from model.monotonic_router import MonotonicFogRouter


def test_router_is_pointwise_monotonic_and_returns_straight_through_hard_route():
    router = MonotonicFogRouter(hidden_channels=4)
    density = torch.tensor([[[[0.0, 0.2], [0.7, 1.0]]]], requires_grad=True)

    output = router(density, temperature=0.7)

    assert set(output) == {"route_logits", "route_soft", "route_hard"}
    assert torch.all((output["route_soft"] >= 0) & (output["route_soft"] <= 1))
    assert torch.all((output["route_hard"] == 0) | (output["route_hard"] == 1))
    ordered_density = torch.linspace(0, 1, 32).view(1, 1, 1, -1)
    ordered_route = router(ordered_density, temperature=1.0)["route_soft"].flatten()
    assert torch.all(ordered_route[1:] >= ordered_route[:-1])

    output["route_hard"].sum().backward()
    assert density.grad is not None
    assert torch.isfinite(density.grad).all()


def test_router_rejects_nonpositive_temperature_and_has_no_neighbor_dependency():
    router = MonotonicFogRouter(hidden_channels=4)
    density = torch.full((1, 1, 3, 3), 0.4)
    changed_neighbor = density.clone()
    changed_neighbor[..., 0, 0] = 1.0

    original = router(density)["route_soft"]
    altered = router(changed_neighbor)["route_soft"]

    assert original[..., 1, 1] == altered[..., 1, 1]
    with pytest.raises(ValueError, match="temperature"):
        router(density, temperature=0.0)
