import torch

from loss.fog_routed_source_loss import (
    masked_mean,
    masked_smooth_l1,
    binary_route_penalty,
    masked_gradient_error,
    masked_local_ssim_error,
)


def test_masked_smooth_l1_uses_effective_support_and_empty_mask_is_zero():
    prediction = torch.tensor([[[[0.0, 2.0]]]])
    target = torch.tensor([[[[0.0, 0.0]]]])
    mask = torch.tensor([[[[1.0, 0.0]]]])

    assert masked_smooth_l1(prediction, target, mask, beta=1.0) == 0.0
    assert masked_mean(prediction, torch.zeros_like(mask)) == 0.0


def test_binary_route_penalty_is_zero_at_extremes_and_pushes_away_from_half():
    route = torch.tensor([[[[0.25, 0.5, 0.75]]]], requires_grad=True)
    penalty = binary_route_penalty(route, torch.ones_like(route))

    assert penalty > 0
    penalty.backward()
    assert route.grad[0, 0, 0, 0] > 0  # gradient descent decreases values below .5
    assert route.grad[0, 0, 0, 2] < 0  # gradient descent increases values above .5
    assert binary_route_penalty(torch.tensor([[[[0.0, 1.0]]]]), torch.ones(1, 1, 1, 2)) == 0.0


def test_support_aware_gradient_and_ssim_do_not_read_outside_region():
    target = torch.zeros(1, 1, 7, 7)
    support = torch.zeros_like(target)
    support[..., 2:5, 2:5] = 1
    prediction_a = target.clone()
    prediction_b = target.clone()
    prediction_b[..., 0, :] = 10
    prediction_b[..., :, 0] = 10

    assert torch.allclose(
        masked_gradient_error(prediction_a, target, support),
        masked_gradient_error(prediction_b, target, support),
    )
    assert torch.allclose(
        masked_local_ssim_error(prediction_a, target, support, window_size=3, min_valid_support=2),
        masked_local_ssim_error(prediction_b, target, support, window_size=3, min_valid_support=2),
    )
