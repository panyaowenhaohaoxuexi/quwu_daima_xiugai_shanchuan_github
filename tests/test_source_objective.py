import torch

from loss.source import compute_physical_mask_losses


def test_physical_mask_objective_supervises_each_mask_pixel_and_backpropagates_logits():
    prediction = torch.zeros(1, 3, 2, 2, requires_grad=True)
    clear = torch.zeros_like(prediction)
    density = torch.zeros(1, 1, 2, 2, requires_grad=True)
    logits = torch.tensor([[[[-4.0, 4.0], [4.0, -4.0]]]], requires_grad=True)
    mask = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])

    losses = compute_physical_mask_losses(prediction, clear, density, torch.zeros_like(density), logits, mask,
                                          lambda_density=2.0, lambda_route=3.0, density_smooth_l1_beta=0.1)

    losses["total"].backward()
    assert set(losses) == {"reconstruction", "density", "route", "total", "pos_weight"}
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert losses["route"] > 0


@torch.no_grad()
def test_physical_mask_objective_is_finite_for_empty_and_full_masks():
    prediction = torch.zeros(1, 3, 2, 2)
    density = torch.zeros(1, 1, 2, 2)
    logits = torch.zeros_like(density)
    for mask in (torch.zeros_like(density), torch.ones_like(density)):
        losses = compute_physical_mask_losses(prediction, prediction, density, density, logits, mask)
        assert torch.isfinite(losses["total"])
        assert 1.0 <= float(losses["pos_weight"]) <= 100.0
