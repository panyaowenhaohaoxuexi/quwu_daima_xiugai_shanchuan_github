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


class IRDifferenceStructureEncoder(nn.Module):
    """Encode IR structure with fixed center, directional, and radial differences."""

    def __init__(self):
        super().__init__()
        cdc_kernel = torch.tensor(
            [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        adc_kernels = torch.tensor(
            [
                [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]],
                [[-1.0, -1.0, -1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            ],
            dtype=torch.float32,
        ).unsqueeze(1)
        rdc_kernels = torch.tensor(
            [
                [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                [[0.0, 0.0, -1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        ).unsqueeze(1)
        self.register_buffer("cdc_kernel", cdc_kernel)
        self.register_buffer("adc_kernels", adc_kernels)
        self.register_buffer("rdc_kernels", rdc_kernels)

        self.cdc_adapter = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.adc_adapter = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.rdc_adapter = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(48, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_ir_01):
        ir_gray = x_ir_01.mean(dim=1, keepdim=True)
        cdc = F.conv2d(ir_gray, self.cdc_kernel, padding=1)
        adc = F.conv2d(ir_gray, self.adc_kernels, padding=1)
        rdc = F.conv2d(ir_gray, self.rdc_kernels, padding=1)
        cdc_feat = self.cdc_adapter(cdc)
        adc_feat = self.adc_adapter(adc)
        rdc_feat = self.rdc_adapter(rdc)
        return self.fuse(torch.cat([cdc_feat, adc_feat, rdc_feat], dim=1))


class HDE(nn.Module):
    """Visible/IR dual-stream haze distribution estimator.

    Visible features drive deformable sampling while fixed-difference IR
    structure supplies an offset guide and a reference for lost structure.

    Inputs:
        x_vis_01: (B, 3, H, W), de-normalized visible image in [0, 1].
        x_ir_01:  (B, 3, H, W), de-normalized infrared image in [0, 1].
    Outputs:
        density_map: (B, 1, H, W), high values indicate dense haze.
        density_feat: (B, 96, H, W), visible-only feature for mask_head.
    """

    def __init__(self):
        super().__init__()
        self.conv_init = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.ir_struct_encoder = IRDifferenceStructureEncoder()
        self.ir_struct_to64 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Offset channels = 2 * 3 * 3 = 18. IR-guided heads consume both streams.
        self.offset_conv1_vis = nn.Conv2d(32, 18, kernel_size=3, padding=1, bias=True)
        self.offset_conv1_ir = nn.Conv2d(64, 18, kernel_size=3, padding=1, bias=True)
        self.deform_conv1 = DeformConv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn_relu1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.offset_conv2_vis = nn.Conv2d(64, 18, kernel_size=3, padding=1, bias=True)
        self.offset_conv2_ir = nn.Conv2d(128, 18, kernel_size=3, padding=1, bias=True)
        self.deform_conv2 = DeformConv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn_relu2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.conv_ms1 = nn.Conv2d(64, 32, kernel_size=1, padding=0, bias=False)
        self.conv_ms2 = nn.Conv2d(64, 32, kernel_size=3, padding=2, dilation=2, bias=False)
        self.conv_ms3 = nn.Conv2d(64, 32, kernel_size=3, padding=4, dilation=4, bias=False)
        self.ir_ms1 = nn.Conv2d(64, 32, kernel_size=1, padding=0, bias=False)
        self.ir_ms2 = nn.Conv2d(64, 32, kernel_size=3, padding=2, dilation=2, bias=False)
        self.ir_ms3 = nn.Conv2d(64, 32, kernel_size=3, padding=4, dilation=4, bias=False)

        self.attn_conv = nn.Conv2d(4, 1, kernel_size=7, padding=3, bias=True)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        for offset_head in (
            self.offset_conv1_vis,
            self.offset_conv1_ir,
            self.offset_conv2_vis,
            self.offset_conv2_ir,
        ):
            nn.init.zeros_(offset_head.weight)
            nn.init.zeros_(offset_head.bias)

    def forward(self, x_vis_01, x_ir_01=None, return_feat=False, return_debug=False):
        if x_ir_01 is None:
            x_ir_01 = x_vis_01

        f_vis = self.conv_init(x_vis_01)
        ir_struct = self.ir_struct_encoder(x_ir_01)
        ir_struct_64 = self.ir_struct_to64(ir_struct)

        offset1_vis = self.offset_conv1_vis(f_vis)
        offset1_ir = self.offset_conv1_ir(torch.cat([f_vis, ir_struct], dim=1))
        f = self.deform_conv1(f_vis, offset1_vis + offset1_ir)
        f = self.bn_relu1(f)

        offset2_vis = self.offset_conv2_vis(f)
        offset2_ir = self.offset_conv2_ir(torch.cat([f, ir_struct_64], dim=1))
        f = self.deform_conv2(f, offset2_vis + offset2_ir)
        f = self.bn_relu2(f)

        fm_vis = torch.cat(
            [self.conv_ms1(f), self.conv_ms2(f), self.conv_ms3(f)],
            dim=1,
        )
        fm_ir = torch.cat(
            [self.ir_ms1(ir_struct_64), self.ir_ms2(ir_struct_64), self.ir_ms3(ir_struct_64)],
            dim=1,
        )

        vis_gap = fm_vis.mean(dim=1, keepdim=True)
        vis_gmp = fm_vis.max(dim=1, keepdim=True).values
        ir_gap = fm_ir.mean(dim=1, keepdim=True)
        ir_gmp = fm_ir.max(dim=1, keepdim=True).values
        struct_diff_gap = torch.abs(vis_gap - ir_gap)
        struct_diff_gmp = torch.abs(vis_gmp - ir_gmp)
        attn_input = torch.cat(
            [vis_gap, vis_gmp, struct_diff_gap, struct_diff_gmp],
            dim=1,
        )
        density_map = torch.sigmoid(self.attn_conv(attn_input))

        if return_debug:
            debug = {
                "ir_struct": ir_struct,
                "fm_vis": fm_vis,
                "fm_ir": fm_ir,
                "struct_diff_gap": struct_diff_gap,
                "struct_diff_gmp": struct_diff_gmp,
                "offset1_vis": offset1_vis,
                "offset1_ir": offset1_ir,
                "offset2_vis": offset2_vis,
                "offset2_ir": offset2_ir,
            }
            if return_feat:
                return density_map, fm_vis, debug
            return density_map, debug
        if return_feat:
            return density_map, fm_vis
        return density_map
