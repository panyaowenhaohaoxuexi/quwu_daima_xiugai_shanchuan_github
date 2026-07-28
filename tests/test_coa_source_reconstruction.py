import pytest
import torch

from loss.source import compute_source_objective


def test_global_reconstruction_uses_coa_weighted_l1_ssim_and_contrast_with_hazy_negative():
    prediction = torch.ones(1, 3, 4, 4, requires_grad=True)
    clear = torch.zeros_like(prediction)
    hazy = torch.full_like(prediction, 0.5)
    density = torch.zeros(1, 1, 4, 4)
    captured = {}

    def coa_ssim(prediction_arg, clear_arg):
        assert prediction_arg is prediction
        assert clear_arg is clear
        return prediction_arg.new_tensor(0.25)

    def coa_contrast(prediction_arg, clear_arg, hazy_arg):
        captured["arguments"] = (prediction_arg, clear_arg, hazy_arg)
        return prediction_arg.mean() * 0 + 0.4

    output = compute_source_objective(
        prediction, clear, density, density, torch.full_like(density, 0.5), torch.zeros_like(density),
        torch.zeros_like(density), torch.zeros_like(density), hazy_rgb=hazy,
        global_ssim_criterion=coa_ssim, global_contrast_criterion=coa_contrast,
        global_l1_weight=0.8, global_ssim_weight=0.2, global_contrast_weight=0.05,
        region_l1_weight=1.0, region_ssim_weight=0.2, region_gradient_weight=0.2,
    )

    assert output["global_l1"].item() == pytest.approx(1.0)
    assert output["global_ssim"].item() == pytest.approx(0.75)
    assert output["global_contrast"].item() == pytest.approx(0.4)
    assert output["global"].item() == pytest.approx(0.97)
    assert captured["arguments"][0] is prediction
    assert captured["arguments"][1] is clear
    assert captured["arguments"][2] is hazy


def test_fuse_completion_and_boundary_share_the_same_region_loss_formula_and_weights():
    prediction = torch.ones(1, 3, 8, 8)
    clear = torch.zeros_like(prediction)
    hazy = torch.full_like(prediction, 0.5)
    density = torch.zeros(1, 1, 8, 8)
    full_region = torch.ones_like(density)
    output = compute_source_objective(
        prediction, clear, density, density, full_region, full_region,
        torch.zeros_like(density), torch.zeros_like(density), hazy_rgb=hazy,
        global_ssim_criterion=lambda _prediction, _clear: prediction.new_tensor(1.0),
        global_contrast_criterion=lambda _prediction, _clear, _hazy: prediction.new_zeros(()),
        region_l1_weight=1.0, region_ssim_weight=0.2, region_gradient_weight=0.2,
    )

    assert output["comp"].item() == pytest.approx(output["boundary"].item())
    assert output["region_l1_weight"] == 1.0
    assert output["region_ssim_weight"] == 0.2
    assert output["region_gradient_weight"] == 0.2
