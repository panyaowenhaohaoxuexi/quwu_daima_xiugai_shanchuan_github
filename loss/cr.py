import torch.nn as nn
import torch
from torch.nn import functional as F
import torch.nn.functional as fnn
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import models
from torchvision.models import VGG19_Weights


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        try:
            vgg_pretrained_features = models.vgg19(weights=VGG19_Weights.DEFAULT).features
        except Exception as exc:
            raise RuntimeError(
                "ContrastLoss requires torchvision VGG19 pretrained weights. "
                "Please pre-cache the weights or disable w_loss_Cr."
            ) from exc
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]


class ContrastLoss(nn.Module):
    def __init__(self, ablation=False):

        super(ContrastLoss, self).__init__()
        self.vgg = Vgg19()
        for p in self.vgg.parameters():
            p.requires_grad = False
        self.l1 = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        self.ab = ablation
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

    def forward(self, a, p, n):
        if next(self.vgg.parameters()).device != a.device:
            self.vgg.to(a.device)
        self.vgg.eval()
        mean = self.mean.to(device=a.device, dtype=a.dtype)
        std = self.std.to(device=a.device, dtype=a.dtype)
        a = (a - mean) / std
        p = (p - mean) / std
        n = (n - mean) / std
        a_vgg, p_vgg, n_vgg = self.vgg(a), self.vgg(p), self.vgg(n)
        loss = 0

        d_ap, d_an = 0, 0
        for i in range(len(a_vgg)):
            d_ap = self.l1(a_vgg[i], p_vgg[i].detach())
            if not self.ab:
                d_an = self.l1(a_vgg[i], n_vgg[i].detach())
                contrastive = d_ap / (d_an + 1e-7)
            else:
                contrastive = d_ap

            loss += self.weights[i] * contrastive
        return loss


class VGGLoss(nn.Module):
    def __init__(self, device, n_layers=5):
        super().__init__()

        feature_layers = (2, 7, 12, 21, 30)
        self.weights = (1.0, 1.0, 1.0, 1.0, 1.0)

        vgg = torchvision.models.vgg19(pretrained=True).features
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))
        self.layers = nn.ModuleList()
        prev_layer = 0
        for next_layer in feature_layers[:n_layers]:
            layers = nn.Sequential()
            for layer in range(prev_layer, next_layer):
                layers.add_module(str(layer), vgg[layer])
            self.layers.append(layers.to(device))
            prev_layer = next_layer

        for param in self.parameters():
            param.requires_grad = False

        self.criterion = nn.L1Loss().to(device)

    def forward(self, source, target):
        mean = self.mean.to(device=source.device, dtype=source.dtype)
        std = self.std.to(device=source.device, dtype=source.dtype)
        source = (source - mean) / std
        target = (target - mean) / std
        loss = 0
        for layer, weight in zip(self.layers, self.weights):
            source = layer(source)
            with torch.no_grad():
                target = layer(target)
            loss += weight * self.criterion(source, target)

        return loss
