import pytest
import torch

from training.metrics import psnr, ssim_global


def test_formal_metrics_are_finite_and_perfect_prediction_has_high_psnr_ssim():
    target = torch.rand(1, 3, 16, 16)

    assert psnr(target, target) > 100
    assert ssim_global(target, target) == pytest.approx(1.0)
    assert torch.isfinite(psnr(target, torch.zeros_like(target)))
    assert torch.isfinite(ssim_global(target, torch.zeros_like(target)))
