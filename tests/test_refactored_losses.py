import torch


def test_refactored_source_and_ema_losses_match_current_formulas():
    from loss.ema import real_consistency_loss as migrated_real_loss
    from loss.real.consistency import real_consistency_loss as legacy_real_loss
    from loss.source import compute_source_objective as migrated_source_loss
    from loss.synthetic.objective import compute_source_objective as legacy_source_loss

    torch.manual_seed(31)
    pred = torch.rand(1, 3, 8, 8, requires_grad=True)
    clear = torch.rand(1, 3, 8, 8)
    density = torch.rand(1, 1, 8, 8, requires_grad=True)
    density_gt = torch.rand(1, 1, 8, 8)
    route = torch.rand(1, 1, 8, 8, requires_grad=True)
    boundary = torch.rand(1, 1, 8, 8)
    q = torch.rand(1, 1, 8, 8)
    support = torch.ones(1, 1, 8, 8)

    legacy = legacy_source_loss(pred, clear, density, density_gt, route, boundary, q, support)
    migrated = migrated_source_loss(pred, clear, density, density_gt, route, boundary, q, support)
    assert legacy.keys() == migrated.keys()
    for key in legacy:
        torch.testing.assert_close(legacy[key], migrated[key], rtol=0, atol=0)

    real_inputs = [torch.rand(1, 1, 8, 8) for _ in range(9)]
    legacy_real = legacy_real_loss(*real_inputs)
    migrated_real = migrated_real_loss(*real_inputs)
    assert legacy_real.keys() == migrated_real.keys()
    for key in legacy_real:
        torch.testing.assert_close(legacy_real[key], migrated_real[key], rtol=0, atol=0)
