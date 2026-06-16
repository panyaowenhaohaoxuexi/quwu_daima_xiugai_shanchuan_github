import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.ops import DeformConv2d as _TorchvisionDeformConv2d
except Exception:
    _TorchvisionDeformConv2d = None


class _FallbackDeformConv2d(nn.Module):
    """Conv2d fallback with the same forward(x, offset) interface."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x, offset):
        return self.conv(x)


DeformConv2d = _TorchvisionDeformConv2d or _FallbackDeformConv2d


class HDE(nn.Module):
    """Haze Distribution Estimator (adopted from HDCFN, ACM MM'25).

    Estimates per-pixel haze density from visible image only,
    using deformable convolutions to adapt to irregular haze
    shapes and multi-scale spatial attention for density output.

    Input:  x_vis_01  (B, 3, H, W)  de-normalized to [0, 1]
    Output: M_vis     (B, 1, H, W)  in [0, 1], high = dense haze
    """

    def __init__(self):
        super().__init__()

        # Stage 1: Initial feature extraction
        self.conv_init = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Stage 2: Deformable conv block 1  32->64
        # offset channels = 2 * kH * kW = 2*3*3 = 18
        self.offset_conv1 = nn.Conv2d(32, 18, kernel_size=3,
                                      padding=1, bias=True)
        self.deform_conv1 = DeformConv2d(32, 64, kernel_size=3,
                                         padding=1, bias=False)
        self.bn_relu1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Stage 3: Deformable conv block 2  64->64
        self.offset_conv2 = nn.Conv2d(64, 18, kernel_size=3,
                                      padding=1, bias=True)
        self.deform_conv2 = DeformConv2d(64, 64, kernel_size=3,
                                         padding=1, bias=False)
        self.bn_relu2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Stage 4: Multi-scale convolutions (3 parallel branches)
        self.conv_ms1 = nn.Conv2d(64, 32, kernel_size=1,
                                  padding=0, bias=False)
        self.conv_ms2 = nn.Conv2d(64, 32, kernel_size=3,
                                  padding=2, dilation=2, bias=False)
        self.conv_ms3 = nn.Conv2d(64, 32, kernel_size=3,
                                  padding=4, dilation=4, bias=False)

        # Stage 5: Spatial attention -> haze density map
        self.attn_conv = nn.Conv2d(2, 1, kernel_size=7,
                                   padding=3, bias=True)

        # Weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # offset convs -> zero init = identity deformation at start
        nn.init.zeros_(self.offset_conv1.weight)
        nn.init.zeros_(self.offset_conv1.bias)
        nn.init.zeros_(self.offset_conv2.weight)
        nn.init.zeros_(self.offset_conv2.bias)

    def forward(self, x_vis_01, return_feat=False):
        # Stage 1
        f = self.conv_init(x_vis_01)            # (B, 32, H, W)

        # Stage 2
        offset1 = self.offset_conv1(f)          # (B, 18, H, W)
        f = self.deform_conv1(f, offset1)       # (B, 64, H, W)
        f = self.bn_relu1(f)

        # Stage 3
        offset2 = self.offset_conv2(f)          # (B, 18, H, W)
        f = self.deform_conv2(f, offset2)       # (B, 64, H, W)
        f = self.bn_relu2(f)

        # Stage 4: multi-scale concat
        f1 = self.conv_ms1(f)                   # (B, 32, H, W)
        f2 = self.conv_ms2(f)                   # (B, 32, H, W)
        f3 = self.conv_ms3(f)                   # (B, 32, H, W)
        fm = torch.cat([f1, f2, f3], dim=1)    # (B, 96, H, W)

        # Stage 5: spatial attention
        gap = fm.mean(dim=1, keepdim=True)      # (B, 1, H, W)
        gmp = fm.max(dim=1, keepdim=True)[0]   # (B, 1, H, W)
        M_vis = torch.sigmoid(
            self.attn_conv(torch.cat([gap, gmp], dim=1))
        )                                        # (B, 1, H, W)
        if return_feat:
            return M_vis, fm
        return M_vis
