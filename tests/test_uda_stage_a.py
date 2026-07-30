import torch

from UDA import route_consistency_multiplier, style_source_batch


def test_style_source_batch_preserves_physical_labels_and_can_style_every_sample():
    source = (
        torch.rand(2, 3, 8, 8),
        torch.rand(2, 3, 8, 8),
        torch.rand(2, 3, 8, 8),
        torch.rand(2, 1, 8, 8),
        torch.randint(0, 2, (2, 1, 8, 8)).float(),
    )
    target_hazy, target_tir = torch.rand(2, 3, 10, 12), torch.rand(2, 3, 10, 12)

    styled, applied = style_source_batch(
        source, target_hazy, target_tir, generator=torch.Generator().manual_seed(3),
        probability=1.0, beta_min=1.0, beta_max=1.0,
        min_gain=0.75, max_gain=1.35, max_abs_bias=0.20,
    )

    assert applied.all()
    assert torch.equal(styled[3], source[3])
    assert torch.equal(styled[4], source[4])
    assert not torch.equal(styled[0], source[0])


def test_route_consistency_multiplier_has_warmup_then_linear_ramp():
    assert route_consistency_multiplier(0, warmup_steps=10, ramp_steps=20) == 0.0
    assert route_consistency_multiplier(10, warmup_steps=10, ramp_steps=20) == 0.0
    assert route_consistency_multiplier(20, warmup_steps=10, ramp_steps=20) == 0.5
    assert route_consistency_multiplier(30, warmup_steps=10, ramp_steps=20) == 1.0
