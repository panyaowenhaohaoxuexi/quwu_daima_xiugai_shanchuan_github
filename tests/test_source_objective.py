import torch

from loss.source import compute_source_objective


def test_source_objective_uses_q_only_inside_omega_and_binary_penalty():
    pred = torch.zeros(1, 3, 2, 2, requires_grad=True)
    clear = torch.zeros_like(pred)
    density = torch.zeros(1, 1, 2, 2, requires_grad=True)
    density_gt = torch.zeros_like(density)
    route = torch.full((1, 1, 2, 2), 0.25, requires_grad=True)
    q = torch.ones_like(route)
    omega = torch.zeros_like(route)
    omega[..., 0, 0] = 1
    output = compute_source_objective(pred, clear, density, density_gt, route, torch.zeros_like(route), q, omega)

    output["total"].backward()
    assert route.grad[..., 0, 0].abs() > 0
    assert route.grad[..., 1, 1].abs() > 0  # binary penalty remains global
    assert torch.isfinite(output["total"])


def test_source_objective_applies_outer_router_weight_to_density_route_and_binary_only():
    tensors = dict(
        pred_clear=torch.ones(1, 3, 2, 2), clear_rgb=torch.zeros(1, 3, 2, 2),
        density_map=torch.ones(1, 1, 2, 2), density_gt=torch.zeros(1, 1, 2, 2),
        route_soft=torch.full((1, 1, 2, 2), 0.25), boundary_map=torch.zeros(1, 1, 2, 2),
        q=torch.ones(1, 1, 2, 2), omega_support=torch.ones(1, 1, 2, 2),
    )
    without_router = compute_source_objective(**tensors, lambda_router=0.0)
    with_router = compute_source_objective(**tensors, lambda_router=1.0)

    assert with_router["total"] > without_router["total"]
