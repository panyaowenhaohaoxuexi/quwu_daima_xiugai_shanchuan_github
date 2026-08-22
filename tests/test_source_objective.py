import torch
import pytest

from loss.source import compute_physical_mask_losses


def test_physical_mask_objective_supervises_each_mask_pixel_and_backpropagates_logits():
    prediction = torch.zeros(1, 3, 2, 2, requires_grad=True)
    clear = torch.zeros_like(prediction)
    density = torch.zeros(1, 1, 2, 2, requires_grad=True)
    logits = torch.tensor([[[[-4.0, 4.0], [4.0, -4.0]]]], requires_grad=True)
    mask = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])

    route = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]], requires_grad=True)
    boundary = torch.zeros_like(route)
    losses = compute_physical_mask_losses(
        prediction, clear, density, torch.zeros_like(density), logits, mask,
        route_for_reconstruction=route, boundary_map=boundary, hazy_rgb=prediction,
        global_ssim_criterion=lambda _pred, _clear: prediction.new_tensor(1.0),
        global_contrast_criterion=lambda _pred, _clear, _hazy: prediction.new_zeros(()),
        lambda_density=2.0, lambda_route=3.0, density_smooth_l1_beta=0.1,
    )

    losses["total"].backward()
    assert {"reconstruction", "global", "fuse", "comp", "boundary", "density", "route", "total", "pos_weight"} <= set(losses)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert route.grad is None  # regional reconstruction must not supervise the router.
    assert losses["route"] > 0


@torch.no_grad()
def test_physical_mask_objective_is_finite_for_empty_and_full_masks():
    prediction = torch.zeros(1, 3, 2, 2)
    density = torch.zeros(1, 1, 2, 2)
    logits = torch.zeros_like(density)
    for mask in (torch.zeros_like(density), torch.ones_like(density)):
        losses = compute_physical_mask_losses(
            prediction, prediction, density, density, logits, mask,
            route_for_reconstruction=density, boundary_map=density, hazy_rgb=prediction,
            global_ssim_criterion=lambda _pred, _clear: prediction.new_tensor(1.0),
            global_contrast_criterion=lambda _pred, _clear, _hazy: prediction.new_zeros(()),
        )
        assert torch.isfinite(losses["total"])
        assert 1.0 <= float(losses["pos_weight"]) <= 100.0


def test_physical_mask_objective_computes_density_only_for_samples_with_density_gt():
    prediction = torch.zeros(2, 3, 2, 2)
    density = torch.ones(2, 1, 2, 2, requires_grad=True)
    density_gt = torch.cat((torch.zeros_like(density[:1]), torch.full_like(density[1:], 100.0)))
    logits = torch.zeros_like(density, requires_grad=True)
    route = torch.zeros_like(density)
    losses = compute_physical_mask_losses(
        prediction, prediction, density, density_gt, logits, torch.zeros_like(density),
        density_valid=torch.tensor([True, False]), route_for_reconstruction=route,
        boundary_map=route, hazy_rgb=prediction,
        global_ssim_criterion=lambda _pred, _clear: prediction.new_tensor(1.0),
        global_contrast_criterion=lambda _pred, _clear, _hazy: prediction.new_zeros(()),
        density_smooth_l1_beta=0.1,
    )

    assert losses["density"].item() == pytest.approx(0.95)
    assert losses["density_valid_samples"].item() == 1


def test_physical_mask_objective_has_zero_density_loss_when_no_density_gt_is_available():
    prediction = torch.zeros(2, 3, 2, 2)
    density = torch.ones(2, 1, 2, 2, requires_grad=True)
    logits = torch.zeros_like(density)
    route = torch.zeros_like(density)
    losses = compute_physical_mask_losses(
        prediction, prediction, density, torch.zeros_like(density), logits, torch.zeros_like(density),
        density_valid=torch.tensor([False, False]), route_for_reconstruction=route,
        boundary_map=route, hazy_rgb=prediction,
        global_ssim_criterion=lambda _pred, _clear: prediction.new_tensor(1.0),
        global_contrast_criterion=lambda _pred, _clear, _hazy: prediction.new_zeros(()),
    )

    losses["density"].backward()
    assert losses["density"].item() == 0.0
    assert torch.equal(density.grad, torch.zeros_like(density.grad))


def test_regional_reconstruction_uses_old_global_fuse_comp_boundary_weights():
    prediction = torch.ones(1, 3, 8, 8)
    target = torch.zeros_like(prediction)
    density = torch.zeros(1, 1, 8, 8)
    route = torch.ones_like(density)
    losses = compute_physical_mask_losses(
        prediction, target, density, density, density, density,
        route_for_reconstruction=route, boundary_map=route, hazy_rgb=target,
        global_ssim_criterion=lambda _pred, _target: prediction.new_tensor(1.0),
        global_contrast_criterion=lambda _pred, _target, _hazy: prediction.new_zeros(()),
    )
    assert losses["global"].item() == pytest.approx(0.8)
    assert losses["fuse"].item() == 0.0
    assert losses["comp"].item() == pytest.approx(1.2, abs=1e-4)
    assert losses["boundary"].item() == pytest.approx(1.2, abs=1e-4)
