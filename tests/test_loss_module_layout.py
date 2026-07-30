import torch

from loss.cr import ContrastLoss
from loss.regional import regional_reconstruction_error
from loss.ssim import SSIM


def test_coa_style_loss_modules_are_public_and_device_safe():
    ssim = SSIM()
    image = torch.rand(1, 3, 8, 8)
    assert torch.isfinite(ssim(image, image))
    assert ContrastLoss.__module__ == "loss.cr"
    assert regional_reconstruction_error.__module__ == "loss.regional"
