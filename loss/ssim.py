"""CoA Gaussian-window SSIM, with device-safe window caching."""

from math import exp

import torch
from torch.nn import functional as F


def gaussian(window_size, sigma):
    values = torch.tensor([exp(-(index - window_size // 2) ** 2 / float(2 * sigma ** 2))
                           for index in range(window_size)], dtype=torch.float32)
    return values / values.sum()


def create_window(window_size, channel, *, device=None, dtype=None):
    one_dimensional = gaussian(window_size, 1.5).unsqueeze(1)
    two_dimensional = one_dimensional.mm(one_dimensional.t()).unsqueeze(0).unsqueeze(0)
    return two_dimensional.expand(channel, 1, window_size, window_size).contiguous().to(device=device, dtype=dtype)


class SSIM(torch.nn.Module):
    """Formula-equivalent replacement for CoA's ``loss/SSIM.py`` module."""

    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size, self.size_average, self.channel = window_size, size_average, 0
        self.register_buffer("window", torch.empty(0))

    def forward(self, img1, img2):
        if img1.shape != img2.shape:
            raise ValueError("SSIM inputs must have matching shapes")
        channel = img1.shape[1]
        if self.channel != channel or self.window.device != img1.device or self.window.dtype != img1.dtype:
            self.window = create_window(self.window_size, channel, device=img1.device, dtype=img1.dtype)
            self.channel = channel
        window = self.window
        mu1 = F.conv2d(img1, window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size // 2, groups=channel)
        mu1_sq, mu2_sq, mu1_mu2 = mu1.square(), mu2.square(), mu1 * mu2
        sigma1_sq = F.conv2d(img1.square(), window, padding=self.window_size // 2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2.square(), window, padding=self.window_size // 2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size // 2, groups=channel) - mu1_mu2
        value = ((2 * mu1_mu2 + 0.01 ** 2) * (2 * sigma12 + 0.03 ** 2) /
                 ((mu1_sq + mu2_sq + 0.01 ** 2) * (sigma1_sq + sigma2_sq + 0.03 ** 2)))
        return value.mean() if self.size_average else value.mean(1).mean(1).mean(1)
