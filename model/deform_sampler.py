"""Modulated deformable convolution with an explicit pure-PyTorch backend."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class ModulatedDeformSampler(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, bias=True, backend="pure_pytorch"):
        super().__init__()
        if backend not in {"pure_pytorch", "torchvision"}:
            raise ValueError("backend must be 'pure_pytorch' or 'torchvision'")
        if kernel_size != 3 or stride != 1 or dilation != 1:
            raise ValueError("v1 backends support only 3x3, stride=1, dilation=1")
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.padding = kernel_size, padding
        self.stride, self.dilation, self.backend = stride, dilation, backend
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(in_channels * kernel_size * kernel_size)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, value, offset, modulation_mask):
        batch, _, height, width = value.shape
        expected = 2 * self.kernel_size * self.kernel_size
        if offset.shape != (batch, expected, height, width):
            raise ValueError("invalid deformable offset shape")
        if modulation_mask.shape != (batch, expected // 2, height, width):
            raise ValueError("invalid modulation mask shape")
        if self.backend == "torchvision":
            # Deliberately imported at execution time: model construction must
            # not probe optional kernels or change device state.
            from torchvision.ops import deform_conv2d
            return deform_conv2d(
                value, offset, self.weight, self.bias, stride=(self.stride, self.stride),
                padding=(self.padding, self.padding), dilation=(self.dilation, self.dilation),
                mask=modulation_mask,
            )
        y, x = torch.meshgrid(torch.arange(height, device=value.device, dtype=value.dtype),
                              torch.arange(width, device=value.device, dtype=value.dtype), indexing="ij")
        output = value.new_zeros(batch, self.out_channels, height, width)
        index = 0
        for ky in range(3):
            for kx in range(3):
                # torchvision deform_conv2d stores every offset pair as
                # (dy, dx), not grid_sample's (x, y) coordinate order.
                dy, dx = offset[:, 2 * index], offset[:, 2 * index + 1]
                # Do not clamp: out-of-image samples must retain the zero
                # padding semantics of torchvision/ordinary convolution.
                gx = (x + kx - 1 + dx) / max(width - 1, 1) * 2 - 1
                gy = (y + ky - 1 + dy) / max(height - 1, 1) * 2 - 1
                sampled = F.grid_sample(value, torch.stack((gx, gy), dim=-1), align_corners=True,
                                        mode="bilinear", padding_mode="zeros")
                sampled = sampled * modulation_mask[:, index:index + 1]
                output = output + torch.einsum("bihw,oi->bohw", sampled, self.weight[:, :, ky, kx])
                index += 1
        return output if self.bias is None else output + self.bias.view(1, -1, 1, 1)
