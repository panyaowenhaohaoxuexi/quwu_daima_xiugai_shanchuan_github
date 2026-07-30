"""CoA VGG19 contrast/CR loss with device-safe buffers."""

import torch
from torch import nn
from torchvision.models import VGG19_Weights, vgg19


class Vgg19(nn.Module):
    """The five feature slices used by CoA's original ContrastLoss."""

    def __init__(self, requires_grad=False):
        super().__init__()
        features = vgg19(weights=VGG19_Weights.DEFAULT).features
        boundaries = ((0, 2), (2, 7), (7, 12), (12, 21), (21, 30))
        self.slices = nn.ModuleList([nn.Sequential(*features[start:end]) for start, end in boundaries])
        if not requires_grad:
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    def forward(self, value):
        outputs = []
        for layer in self.slices:
            value = layer(value)
            outputs.append(value)
        return outputs


class ContrastLoss(nn.Module):
    """CoA's contrastive ratio ``d(anchor, positive) / d(anchor, negative)``."""

    def __init__(self, ablation=False):
        super().__init__()
        self.vgg, self.l1 = Vgg19(), nn.L1Loss()
        self.weights, self.ablation = (1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0), bool(ablation)
        self.register_buffer("mean", torch.tensor((0.485, 0.456, 0.406)).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor((0.229, 0.224, 0.225)).view(1, -1, 1, 1))

    def forward(self, anchor, positive, negative):
        anchor, positive, negative = ((value - self.mean) / self.std for value in (anchor, positive, negative))
        anchor_features, positive_features, negative_features = self.vgg(anchor), self.vgg(positive), self.vgg(negative)
        loss = anchor.new_zeros(())
        for weight, anchor_feature, positive_feature, negative_feature in zip(
                self.weights, anchor_features, positive_features, negative_features):
            d_ap = self.l1(anchor_feature, positive_feature.detach())
            loss = loss + weight * (d_ap if self.ablation else d_ap / (self.l1(anchor_feature, negative_feature.detach()) + 1e-7))
        return loss
