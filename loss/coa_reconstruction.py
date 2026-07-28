"""CoA-equivalent global SSIM and VGG-19 contrast reconstruction losses."""

from math import exp

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import VGG19_Weights, vgg19


def _gaussian(window_size, sigma):
    values = torch.tensor([exp(-(index - window_size // 2) ** 2 / float(2 * sigma ** 2))
                           for index in range(window_size)])
    return values / values.sum()


def _window(window_size, channel, *, device, dtype):
    kernel = _gaussian(window_size, 1.5).unsqueeze(1)
    kernel = kernel.mm(kernel.t()).unsqueeze(0).unsqueeze(0)
    return kernel.expand(channel, 1, window_size, window_size).contiguous().to(device=device, dtype=dtype)


class CoASSIM(nn.Module):
    """Numerically equivalent to CoA's Gaussian-window SSIM implementation."""

    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size

    def forward(self, first, second):
        if first.shape != second.shape:
            raise ValueError("CoA SSIM inputs must have matching shapes")
        channels = first.shape[1]
        window = _window(self.window_size, channels, device=first.device, dtype=first.dtype)
        mean_first = F.conv2d(first, window, padding=self.window_size // 2, groups=channels)
        mean_second = F.conv2d(second, window, padding=self.window_size // 2, groups=channels)
        mean_first_sq, mean_second_sq = mean_first.square(), mean_second.square()
        covariance_mean = mean_first * mean_second
        variance_first = F.conv2d(first.square(), window, padding=self.window_size // 2, groups=channels) - mean_first_sq
        variance_second = F.conv2d(second.square(), window, padding=self.window_size // 2, groups=channels) - mean_second_sq
        covariance = F.conv2d(first * second, window, padding=self.window_size // 2, groups=channels) - covariance_mean
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        return (((2 * covariance_mean + c1) * (2 * covariance + c2)) /
                ((mean_first_sq + mean_second_sq + c1) * (variance_first + variance_second + c2))).mean()


class _Vgg19Features(nn.Module):
    """The frozen five-slice VGG-19 feature extractor used by CoA ContrastLoss."""

    def __init__(self):
        super().__init__()
        features = vgg19(weights=VGG19_Weights.DEFAULT).features
        self.slices = nn.ModuleList([
            nn.Sequential(*features[0:2]), nn.Sequential(*features[2:7]), nn.Sequential(*features[7:12]),
            nn.Sequential(*features[12:21]), nn.Sequential(*features[21:30]),
        ])
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def forward(self, value):
        outputs = []
        for layer in self.slices:
            value = layer(value)
            outputs.append(value)
        return outputs


class CoAContrastLoss(nn.Module):
    """CoA's VGG-19 positive/negative contrastive reconstruction objective."""

    def __init__(self):
        super().__init__()
        self.vgg = _Vgg19Features()
        self.l1 = nn.L1Loss()
        self.weights = (1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0)
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, -1, 1, 1))

    def forward(self, anchor, positive, negative):
        anchor = (anchor - self.mean) / self.std
        positive = (positive - self.mean) / self.std
        negative = (negative - self.mean) / self.std
        anchor_features, positive_features, negative_features = self.vgg(anchor), self.vgg(positive), self.vgg(negative)
        loss = anchor.new_zeros(())
        for weight, anchor_feature, positive_feature, negative_feature in zip(
            self.weights, anchor_features, positive_features, negative_features,
        ):
            d_ap = self.l1(anchor_feature, positive_feature.detach())
            d_an = self.l1(anchor_feature, negative_feature.detach())
            loss = loss + weight * d_ap / (d_an + 1e-7)
        return loss
