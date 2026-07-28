"""CoA-compatible Res2Net-101 RGB feature encoder."""

import math
from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn


class Bottle2neck(nn.Module):
    """The Res2Net bottleneck used by CoA's ImageNet encoder."""

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, base_width=26, scale=4, stage_type="normal"):
        super().__init__()
        width = int(math.floor(planes * (base_width / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)
        self.nums = 1 if scale == 1 else scale - 1
        if stage_type == "stage":
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        self.convs = nn.ModuleList([
            nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
            for _ in range(self.nums)
        ])
        self.bns = nn.ModuleList([nn.BatchNorm2d(width) for _ in range(self.nums)])
        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stage_type = stage_type
        self.scale = scale
        self.width = width

    def forward(self, value):
        residual = value
        value = self.relu(self.bn1(self.conv1(value)))
        split = torch.split(value, self.width, dim=1)
        for index in range(self.nums):
            current = split[index] if index == 0 or self.stage_type == "stage" else current + split[index]
            current = self.relu(self.bns[index](self.convs[index](current)))
            value = current if index == 0 else torch.cat((value, current), dim=1)
        if self.scale != 1 and self.stage_type == "normal":
            value = torch.cat((value, split[self.nums]), dim=1)
        elif self.scale != 1 and self.stage_type == "stage":
            value = torch.cat((value, self.pool(split[self.nums])), dim=1)
        value = self.bn3(self.conv3(value))
        if self.downsample is not None:
            residual = self.downsample(residual)
        return self.relu(value + residual)


class _Res2NetBase(nn.Module):
    def __init__(self, layers, *, include_layer4, num_classes=1000, base_width=26, scale=4):
        super().__init__()
        self.inplanes = 64
        self.base_width = base_width
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False),
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        if include_layer4:
            self.layer4 = self._make_layer(512, layers[3], stride=2)
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(512 * Bottle2neck.expansion, num_classes)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * Bottle2neck.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * Bottle2neck.expansion, kernel_size=1, bias=False),
                nn.BatchNorm2d(planes * Bottle2neck.expansion),
            )
        layers = [Bottle2neck(self.inplanes, planes, stride, downsample, self.base_width, self.scale, "stage")]
        self.inplanes = planes * Bottle2neck.expansion
        layers.extend(Bottle2neck(self.inplanes, planes, base_width=self.base_width, scale=self.scale)
                      for _ in range(1, blocks))
        return nn.Sequential(*layers)


class Pre_Res2Net(_Res2NetBase):
    """Full CoA/ImageNet network, used only to match the pretrained state dict."""

    def __init__(self, layers=(3, 4, 23, 3), base_width=26, scale=4, num_classes=1000):
        super().__init__(layers, include_layer4=True, num_classes=num_classes, base_width=base_width, scale=scale)


class Res2Net(_Res2NetBase):
    """CoA's truncated h2/h4/h8/h16 Res2Net feature encoder."""

    def __init__(self, layers=(3, 4, 23, 3), base_width=26, scale=4):
        super().__init__(layers, include_layer4=False, base_width=base_width, scale=scale)

    def forward(self, value):
        value = self.relu(self.bn1(self.conv1(value)))
        h2 = value
        value = self.maxpool(value)
        h4 = self.layer1(value)
        h8 = self.layer2(h4)
        h16 = self.layer3(h8)
        return {"h2": h2, "h4": h4, "h8": h8, "h16": h16}


class CoARes2NetRGBEncoder(nn.Module):
    """Load CoA's ImageNet Res2Net-101 weights and project its RGB pyramid."""

    pretrained_path = (Path(__file__).resolve().parent / "imagenet_model" / "res2net101_v1b_26w_4s-0812c246.pth").resolve()

    def __init__(self, base_channels):
        super().__init__()
        self.encoder = Res2Net()
        self._load_coa_pretraining()
        self.widths = (base_channels, base_channels * 2, base_channels * 3, base_channels * 4)
        self.projections = nn.ModuleDict({
            name: nn.Conv2d(source_channels, target_channels, 1)
            for name, source_channels, target_channels in zip(
                ("h2", "h4", "h8", "h16"), (64, 256, 512, 1024), self.widths,
            )
        })

    def _load_coa_pretraining(self):
        if not self.pretrained_path.is_file():
            raise FileNotFoundError(f"CoA Res2Net-101 pretrained weights are missing: {self.pretrained_path}")
        state = torch.load(self.pretrained_path, map_location="cpu")
        if not isinstance(state, Mapping):
            raise TypeError(f"CoA Res2Net-101 weights must be a state dictionary: {self.pretrained_path}")
        reference = Pre_Res2Net()
        try:
            reference.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            raise RuntimeError(f"CoA Res2Net-101 weights are incompatible: {self.pretrained_path}") from exc
        target_state = self.encoder.state_dict()
        matching = {name: value for name, value in reference.state_dict().items() if name in target_state}
        missing = sorted(set(target_state) - set(matching))
        if missing:
            raise RuntimeError("CoA Res2Net-101 weights do not cover RGB encoder keys: " + ", ".join(missing))
        self.encoder.load_state_dict(matching, strict=True)

    def forward(self, value):
        return {name: self.projections[name](feature) for name, feature in self.encoder(value).items()}
