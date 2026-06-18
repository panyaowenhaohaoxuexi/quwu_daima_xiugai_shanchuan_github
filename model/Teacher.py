import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import sys
import torchvision.transforms.functional as TF
# --- [修改] 使用绝对导入（因为我们添加了项目根目录到 sys.path） ---
if __package__:
    from .vifnet_basic_modules import Encoder_B, Decoder_B, Conv_B, CPAB
    from .dsfe import DSFE
    from .cmdn import CMDN
    from .hde import HDE
    from .gumbel_sigmoid import GumbelSigmoidBinarizer
    from .teacher_color import CrossModalSemanticColorTransport
    from .teacher_semantic import SharedSemanticProjection
    from .teacher_fusion import BiDirectionalSemanticFusion
else:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
    for path in (CURRENT_DIR, PROJECT_ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)
    from vifnet_basic_modules import Encoder_B, Decoder_B, Conv_B, CPAB
    from dsfe import DSFE
    from cmdn import CMDN
    from hde import HDE
    from gumbel_sigmoid import GumbelSigmoidBinarizer
    from teacher_color import CrossModalSemanticColorTransport
    from teacher_semantic import SharedSemanticProjection
    from teacher_fusion import BiDirectionalSemanticFusion
# [已移除] CLIP 导入 — 颜色恢复改为纯图内 Cross-Attention


# --- [修改结束] ---

# --- VIFNet 不一致性函数 f(x, y) (保持不变) ---
def f(x, y):
    """VIFNet inconsistency function (code version)"""
    return (1 - x) * (1 - y) + 1 / 2 * x * y


# --- 基础模块 (SobelEdgeDetector, Pre_Res2Net, Bottle2neck, Res2Net(3通道输入), ConvBlock, DeconvBlock, Decoder_MDCBlock1, make_dense, RDB, ConvLayer, UpsampleConvLayer, ResidualBlock) ---
# ... (这些基础模块的代码与上一个版本相同，确保 Res2Net 输入为 3 通道，这里省略以保持简洁) ...
# --- [修改] 替换 SobelEdgeDetector 为 CannyEdgeDetector ---
class CannyEdgeDetector(nn.Module):
    """
    可微的 Canny 边缘检测器 (Soft Canny / Gradient Magnitude)。
    包含：高斯模糊 -> Sobel 梯度计算 -> 梯度幅值。
    为了保持训练时的可微性，这里省略了非极大值抑制(NMS)和双阈值硬截断。
    这种"软边缘"非常适合作为 Dice Loss 或 L1 Loss 的输入。
    """

    def __init__(self, kernel_size=5, sigma=1.0):
        super(CannyEdgeDetector, self).__init__()

        # 1. 生成高斯核 (用于降噪)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (kernel_size - 1) / 2.
        variance = sigma ** 2.

        # 计算高斯分布
        gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                          torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

        # 归一化
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
        self.gaussian_filter = gaussian_kernel.view(1, 1, kernel_size, kernel_size)

        # 2. 定义 Sobel 梯度算子
        self.sobel_filter_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_filter_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        # 将核注册为参数但不计算梯度 (固定权重)
        self.gaussian_filter = nn.Parameter(self.gaussian_filter, requires_grad=False)
        self.sobel_filter_x = nn.Parameter(self.sobel_filter_x, requires_grad=False)
        self.sobel_filter_y = nn.Parameter(self.sobel_filter_y, requires_grad=False)

    def forward(self, x):
        # 确保输入是单通道灰度图
        if x.shape[1] == 3:
            # RGB 转灰度: 0.299*R + 0.587*G + 0.114*B
            x = x[:, 0:1, :, :] * 0.299 + x[:, 1:2, :, :] * 0.587 + x[:, 2:3, :, :] * 0.114
        elif x.shape[1] != 1:
            # 如果不是1或3通道，默认取第一个通道处理
            x = x[:, 0:1, :, :]

        # 1. 高斯模糊 (降噪)
        # padding = kernel_size // 2
        padding = self.gaussian_filter.shape[2] // 2
        x_blurred = F.conv2d(x, self.gaussian_filter, padding=padding, groups=1)

        # 2. 计算梯度
        grad_x = F.conv2d(x_blurred, self.sobel_filter_x, padding=1)
        grad_y = F.conv2d(x_blurred, self.sobel_filter_y, padding=1)

        # 3. 计算梯度幅值 (Soft Edge Map)
        # 加上一个极小值防止 sqrt(0) 梯度为 NaN
        magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)

        # 可选：归一化到 [0, 1] 以更好地配合 Dice Loss，但梯度幅值本身也有意义
        # 这里保持原始幅值，如果 DiceLoss 需要 0-1，可以在外部做 sigmoid 或归一化
        return magnitude


# --- [修改结束] ---

# --- [修改1：新增 CBAM 及融合模块] ---
class ChannelAttentionModule(nn.Module):
    """ 通道注意力模块 (CBAM中的C) """

    def __init__(self, channel, reduction=16):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttentionModule(nn.Module):
    """ 空间注意力模块 (CBAM中的S) """

    def __init__(self, kernel_size=7):
        super(SpatialAttentionModule, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)


class AttentionFusionBlock(nn.Module):
    """
    [核心模块]：用于早期融合的注意力块
    通过 (Vis + IR) -> ChannelAttn -> SpatialAttn 来融合
    """

    def __init__(self, channel, reduction=16, kernel_size=7):
        super(AttentionFusionBlock, self).__init__()
        self.channel_attn = ChannelAttentionModule(channel, reduction)
        self.spatial_attn = SpatialAttentionModule(kernel_size)

    def forward(self, x_vis, x_ir):
        # 1. 简单的特征融合（逐元素相加）
        x_fused_base = x_vis + x_ir

        # 2. 应用 CBAM
        x_fused_ca = self.channel_attn(x_fused_base) * x_fused_base
        x_fused_csa = self.spatial_attn(x_fused_ca) * x_fused_ca

        # 3. 返回融合并增强后的特征 (并加上残差)
        return x_fused_csa + x_fused_base


# --- [修改1：结束] ---


class Pre_Res2Net(nn.Module):
    """
    Pre_Res2Net: 用于加载 ImageNet 预训练权重的 Res2Net 模型结构。
    说明：
    - 这是 Res2Net 模型的一个变体，主要用于加载预训练权重，作为去雾模型的编码器主干。
    - 包含完整的 Res2Net-101 结构（包括分类头），但在去雾任务中通常只使用其特征提取部分。
    - 由初始卷积层、最大池化层、四个阶段的 Bottle2neck 块以及分类头组成。
    - 初始化时会自动为卷积和批归一化层设置权重。
    """

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=1000):
        """
        初始化 Pre_Res2Net 模型。
        参数：
        - block: Bottle2neck 类，定义 Res2Net 的基本构建块。
        - layers: 列表，指定每个阶段的 Bottle2neck 块数量，例如 [3, 4, 23, 3]。
        - baseWidth: 控制 Bottle2neck 中每组通道的基础宽度，默认 26。
        - scale: 控制 Bottle2neck 中特征图的分组数量（多尺度特性），默认 4。
        - num_classes: 分类头的输出类别数，默认 1000（适用于 ImageNet）。
        """
        self.inplanes = 64
        super(Pre_Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        # 初始卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),  # *** 注意这里的输入通道是 3 ***
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Res2Net的四个阶段
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 输出通道为512
        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        构建 Res2Net 的一个阶段（layer），包含多个 Bottle2neck 块。
        参数：
        - block: Bottle2neck 类。
        - planes: 该阶段的基础通道数。
        - blocks: 该阶段的 Bottle2neck 块数量。
        - stride: 第一个块的步长，用于控制下采样。
        返回：
        - nn.Sequential: 包含所有 Bottle2neck 块的序列。
        """
        downsample = None
        # 处理下采样（当步长不为1或输入输出通道数变化时）
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # 添加第一个block（可能包含下采样）
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        # 添加剩余的blocks
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        # --- DEBUG: 打印张量尺寸 ---
        """
                前向传播，输出分类结果（用于 ImageNet 预训练）。
                参数：
                - x: 输入张量，形状为 (batch_size, 3, H, W)。
                返回：
                - x: 分类结果，形状为 (batch_size, num_classes)。
        """
        # print(f'input={x.size()}') # 注释掉调试打印
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # print(f'after maxpool: {x.size()}')

        x = self.layer1(x)
        # print(f'after layer1: {x.size}')
        x = self.layer2(x)
        # print(f'after layer2: {x.size}')
        x = self.layer3(x)
        # print(f'after layer3: {x.size}')
        x = self.layer4(x)
        # print(f'after layer4: {x.size}')

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # print(f'x: {x.size}')
        x = self.fc(x)

        # print(f'after fc output: {x.size}')
        # -----------------------------
        return x


class Bottle2neck(nn.Module):
    """
    Bottle2neck: Res2Net 的核心瓶颈块，引入多尺度特征处理。
    说明：
    - 通过将特征图分成 scale 组，并以级联方式处理，增强多尺度特征表达。
    - 每个块包含 1x1 卷积（降维）、多组 3x3 卷积（多尺度处理）、1x1 卷积（升维）以及残差连接。
    - expansion=4，表示输出通道数是 planes 的 4 倍。
    """
    expansion = 4  # 输出通道相对于planes的扩展倍数

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        """
                初始化 Bottle2neck 块。
                参数：
                - inplanes: 输入通道数。
                - planes: 基础输出通道数（实际输出为 planes * expansion）。
                - stride: 3x3 卷积的步长，控制下采样。
                - downsample: 下采样层，用于调整残差连接的通道数和分辨率。
                - baseWidth: 控制每组通道的基础宽度，默认 26。
                - scale: 特征图分组数量，默认 4。
                - stype: 块类型，'stage' 表示阶段的第一个块，可能需要池化。
        """

        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth / 64.0)))  # 计算基础宽度
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)  # 1x1 卷积
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1  # 需要进行3x3卷积的组数

        # 'stage'类型表示这是每个layer的第一个block，可能需要处理步长
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)

        convs = []  # 存储3x3卷积层
        bns = []  # 存储对应的BN层
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)  # 1x1 卷积，恢复通道数
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample  # 残差连接的下采样层
        self.stype = stype
        self.scale = scale
        self.width = width  # 每组的宽度（通道数）

    def forward(self, x):
        """
                前向传播，处理输入特征并输出多尺度融合结果。
                参数：
                - x: 输入张量，形状为 (batch_size, inplanes, H, W)。
                返回：
                - out: 输出张量，形状为 (batch_size, planes * expansion, H', W')。
        """

        residual = x  # 保存残差连接的输入

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)  # 沿通道维度将特征图分成 'scale' 组
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]  # 第一组或'stage'类型，直接使用
            else:
                sp = sp + spx[i]  # 后续组，与前一组的输出相加
            sp = self.convs[i](sp)
            sp = self.relu(self.bns[i](sp))
            if i == 0:
                out = sp  # 第一个3x3卷积的输出
            else:
                out = torch.cat((out, sp), 1)  # 将后续输出在通道上拼接

        # 处理最后一组（第'scale'组）
        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)  # 'normal'类型，直接拼接
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)  # 'stage'类型，先池化再拼接

        out = self.conv3(out)  # 1x1 卷积
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)  # 对残差输入进行下采样

        out += residual  # 添加残差
        out = self.relu(out)

        return out


class Res2Net(nn.Module):
    """
        Res2Net: 去雾模型的编码器部分，基于 Res2Net 结构。
        *** [区域补全范式]：M=1 纯IR补全，M=0 非对称残差融合 ***
    """

    def __init__(self, block, layers, baseWidth=26, scale=4, in_channels=3):  # 添加 in_channels 参数
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        # 使用 in_channels 参数
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False),  # *** 使用 in_channels ***
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)

        # --- Pass 2 injection adapters for full IR content features ---
        # ir_feat_list order: [H/16(1024ch), H/8(512ch), H/4(256ch), H/2(64ch)]
        self.inject_conv0 = nn.Sequential(  # H/2: 64 -> 64
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.inject_conv1 = nn.Sequential(  # H/4: 256 -> 256
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.inject_conv2 = nn.Sequential(  # H/8: 512 -> 512
            nn.Conv2d(512, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.inject_conv3 = nn.Sequential(  # H/16: 1024 -> 1024
            nn.Conv2d(1024, 1024, kernel_size=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.align_conv0 = nn.Conv2d(64, 64, kernel_size=1)
        self.align_conv2 = nn.Conv2d(512, 512, kernel_size=1)
        self.align_conv3 = nn.Conv2d(1024, 1024, kernel_size=1)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        # ... (make_layer 定义保持不变) ...
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def _density_guided_beta(self, vis_feat, ir_used, density_map, fusion_head, expected_channels):
        density_s = F.interpolate(density_map, size=vis_feat.shape[2:], mode='bilinear', align_corners=False)
        fusion_input = torch.cat([vis_feat, ir_used, density_s], dim=1)
        assert fusion_input.shape[1] == expected_channels, (
            f"fusion head channel mismatch: expected {expected_channels}, got {fusion_input.shape[1]}"
        )
        beta = fusion_head(fusion_input)
        assert beta.shape[1] == 1, f"fusion beta should be single-channel, got {beta.shape[1]}"
        return beta

    def forward(self, x, ir_feat_list=None, beta_list=None, haze_mask=None,
                density_map=None, fusion_weight_heads=None, bidir_fusion=None,
                return_region_debug=False):
        """
        [区域补全范式] Res2Net forward，按 HAPM 掩码严格分流：
          - M=1（补全区）：完整、未对齐的 IR 内容特征替代 VIS 特征
          - M=0（融合区）：H/4 双向语义检索，其余尺度注入 IR 结构残差

        ir_feat_list: 纯 IR 内容特征，来自 encoder_ir 的完整多尺度编码特征。
                      顺序为 [H/16, H/8, H/4]，通道分别为 [1024, 512, 256]。
        beta_list:    backward-compatible fallback. New Teacher passes density_map
                      and fusion_weight_heads instead.
        haze_mask:    (B,1,H,W) 掩码，M=1 表示补全区域，M=0 表示融合区域。
                      外部传入时应已为二值或 straight-through binary-like；
                      严格替代只在 mask 值恰为 1 的位置保证。

        ir_feat_list=None 时为纯特征提取模式（IR 并行编码器使用，不做注入）
        """

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x_layer0 = x  # (B, 64, H/2, W/2) - 用于 H4 蒸馏 和 H/2 解码
        x_maxpool = self.maxpool(x)  # (B, 64, H/4, W/4)

        # --- 纯特征提取模式（IR 编码器，不做注入）---
        if ir_feat_list is None:
            x_layer1_orig = self.layer1(x_maxpool)    # (B, 256, H/4, W/4)
            x_layer2_orig = self.layer2(x_layer1_orig) # (B, 512, H/8, W/8)
            x_layer3_orig = self.layer3(x_layer2_orig) # (B, 1024, H/16, W/16)
            fused_outputs = [x_layer3_orig, x_layer2_orig, x_layer1_orig, x_layer0]
            original_outputs = [x_layer3_orig, x_layer2_orig, x_layer1_orig, x_layer0]
            return fused_outputs, original_outputs

        assert ir_feat_list[0].shape[1] == 1024, \
            f"H/16 IR content feature should have 1024 channels, got {ir_feat_list[0].shape[1]}"
        assert ir_feat_list[1].shape[1] == 512, \
            f"H/8 IR content feature should have 512 channels, got {ir_feat_list[1].shape[1]}"
        assert ir_feat_list[2].shape[1] == 256, \
            f"H/4 IR content feature should have 256 channels, got {ir_feat_list[2].shape[1]}"

        if fusion_weight_heads is None and beta_list is None:
            raise ValueError("Res2Net injection requires fusion_weight_heads+density_map or beta_list.")

        fused_debug = {}

        # --- H/2 shallow feature protection (conv1 output, 64ch) ---
        if len(ir_feat_list) >= 4:
            F_ir_0 = self.inject_conv0(ir_feat_list[3])
            F_ir_0 = F.interpolate(F_ir_0, size=x_layer0.shape[2:], mode='bilinear', align_corners=False)
            F_ir_0_aligned = self.align_conv0(F_ir_0)
            if fusion_weight_heads is not None:
                if density_map is None:
                    raise ValueError("density_map is required when using fusion_weight_heads.")
                g_0 = self._density_guided_beta(
                    x_layer0, F_ir_0_aligned, density_map, fusion_weight_heads[3], 129
                )
            else:
                g_0 = F.interpolate(beta_list[3], size=x_layer0.shape[2:], mode='bilinear', align_corners=False)
            if haze_mask is not None:
                mask_0 = F.interpolate(haze_mask, size=x_layer0.shape[2:], mode='nearest')
            else:
                mask_0 = torch.zeros_like(g_0)
            x_layer0_safe = (1.0 - mask_0) * (x_layer0 + g_0 * F_ir_0_aligned) + mask_0 * F_ir_0
        else:
            F_ir_0 = x_layer0
            g_0 = torch.zeros(x_layer0.shape[0], 1, x_layer0.shape[2], x_layer0.shape[3],
                              device=x_layer0.device, dtype=x_layer0.dtype)
            x_layer0_safe = x_layer0

        # --- H/4 尺度 (layer1, 256ch) ---
        x_layer1_orig = self.layer1(x_maxpool)  # (B, 256, H/4, W/4)
        F_vis_1 = x_layer1_orig
        F_ir_1 = self.inject_conv1(ir_feat_list[2])  # H/4, 256->256, IR content
        F_ir_1 = F.interpolate(F_ir_1, size=F_vis_1.shape[2:], mode='bilinear', align_corners=False)
        if haze_mask is not None:
            mask_1 = F.interpolate(haze_mask, size=F_vis_1.shape[2:], mode='nearest')
        else:
            mask_1 = torch.zeros(F_vis_1.shape[0], 1, F_vis_1.shape[2], F_vis_1.shape[3],
                                 device=F_vis_1.device, dtype=F_vis_1.dtype)
        if bidir_fusion is not None and fusion_weight_heads is not None:
            x_layer1_fused, fusion_debug_h4 = bidir_fusion(
                F_vis=F_vis_1,
                F_ir=F_ir_1,
                density_map=density_map,
                mask=mask_1,
                fusion_head=fusion_weight_heads[2],
            )
            beta_1 = fusion_debug_h4["g"]
        else:
            if fusion_weight_heads is not None:
                beta_1 = self._density_guided_beta(
                    F_vis_1, F_ir_1, density_map, fusion_weight_heads[2], 513
                )
            else:
                beta_1 = F.interpolate(beta_list[2], size=F_vis_1.shape[2:], mode='bilinear', align_corners=False)
            beta_eff_1 = mask_1 + (1.0 - mask_1) * beta_1
            x_layer1_fused = (1.0 - beta_eff_1) * F_vis_1 + beta_eff_1 * F_ir_1
            fusion_debug_h4 = None

        # --- H/8 尺度 (layer2, 512ch) ---
        x_layer2_orig = self.layer2(x_layer1_fused)  # (B, 512, H/8, W/8)
        F_vis_2 = x_layer2_orig
        F_ir_2 = self.inject_conv2(ir_feat_list[1])  # H/8, 512->512, IR content
        F_ir_2 = F.interpolate(F_ir_2, size=F_vis_2.shape[2:], mode='bilinear', align_corners=False)
        F_ir_2_aligned = self.align_conv2(F_ir_2)
        if fusion_weight_heads is not None:
            g_2 = self._density_guided_beta(F_vis_2, F_ir_2_aligned, density_map, fusion_weight_heads[1], 1025)
        else:
            g_2 = F.interpolate(beta_list[1], size=F_vis_2.shape[2:], mode='bilinear', align_corners=False)
        if haze_mask is not None:
            mask_2 = F.interpolate(haze_mask, size=F_vis_2.shape[2:], mode='nearest')
        else:
            mask_2 = torch.zeros_like(g_2)
        x_layer2_fused = (1.0 - mask_2) * (F_vis_2 + g_2 * F_ir_2_aligned) + mask_2 * F_ir_2

        # --- H/16 尺度 (layer3, 1024ch) ---
        x_layer3_orig = self.layer3(x_layer2_fused)  # (B, 1024, H/16, W/16)
        F_vis_3 = x_layer3_orig
        F_ir_3 = self.inject_conv3(ir_feat_list[0])  # H/16, 1024->1024, IR content
        F_ir_3 = F.interpolate(F_ir_3, size=F_vis_3.shape[2:], mode='bilinear', align_corners=False)
        F_ir_3_aligned = self.align_conv3(F_ir_3)
        if fusion_weight_heads is not None:
            g_3 = self._density_guided_beta(F_vis_3, F_ir_3_aligned, density_map, fusion_weight_heads[0], 2049)
        else:
            g_3 = F.interpolate(beta_list[0], size=F_vis_3.shape[2:], mode='bilinear', align_corners=False)
        if haze_mask is not None:
            mask_3 = F.interpolate(haze_mask, size=F_vis_3.shape[2:], mode='nearest')
        else:
            mask_3 = torch.zeros_like(g_3)
        x_layer3_fused = (1.0 - mask_3) * (F_vis_3 + g_3 * F_ir_3_aligned) + mask_3 * F_ir_3

        # 返回 注入后(fused)的特征（解码用）和 注入前(orig)的特征（蒸馏用）
        # H/2 uses x_layer0_safe to prevent visible shallow-feature leakage in completion regions.
        fused_outputs = [x_layer3_fused, x_layer2_fused, x_layer1_fused, x_layer0_safe]
        original_outputs = [x_layer3_orig, x_layer2_orig, x_layer1_orig, x_layer0]

        if return_region_debug:
            fused_debug = {
                "fused_feats": [x_layer3_fused, x_layer2_fused, x_layer1_fused, x_layer0_safe],
                "ir_feats": [F_ir_3, F_ir_2, F_ir_1, F_ir_0],
                "vis_feats": [F_vis_3, F_vis_2, F_vis_1, x_layer0],
                "fusion_weights": [g_3, g_2, beta_1, g_0],
                "fusion_debug_h4": fusion_debug_h4,
            }
            return fused_outputs, original_outputs, fused_debug

        return fused_outputs, original_outputs


class ConvBlock(torch.nn.Module):
    # ... (ConvBlock 代码保持不变) ...
    """
        ConvBlock: 标准卷积块，包含卷积、归一化和激活函数。
        说明：
        - 用于解码器中的特征处理，支持多种归一化和激活函数。
        - 结构：Conv2d -> (可选)Norm -> (可选)Activation。
    """

    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        """
                初始化 ConvBlock。
                参数：
                - input_size: 输入通道数。
                - output_size: 输出通道数。
                - kernel_size: 卷积核大小，默认 3。
                - stride: 卷积步长，默认 1。
                - padding: 卷积填充，默认 1。
                - bias: 是否使用偏置，默认 True。
                - activation: 激活函数类型（'relu', 'prelu', 'lrelu', 'tanh', 'sigmoid', 'no'），默认 'prelu'。
                - norm: 归一化类型（'batch', 'instance', None），默认 None。
        """
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        """
                前向传播，执行卷积、归一化和激活操作。
                参数：
                - x: 输入张量，形状为 (batch_size, input_size, H, W)。
                返回：
                - out: 输出张量，形状为 (batch_size, output_size, H', W')。
        """
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation != 'no':
            return self.act(out)
        else:
            return out


class DeconvBlock(torch.nn.Module):
    # ... (DeconvBlock 代码保持不变) ...
    """
        DeconvBlock: 标准转置卷积（反卷积）块，用于上采样。
        说明：
        - 用于解码器中的上采样操作，支持多种归一化和激活函数。
        - 结构：ConvTranspose2d -> (可选)Norm -> (可选)Activation。
    """

    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu',
                 norm=None):
        """
                初始化 DeconvBlock。
                参数：
                - input_size: 输入通道数。
                - output_size: 输出通道数。
                - kernel_size: 转置卷积核大小，默认 4。
                - stride: 转置卷积步长，默认 2（上采样）。
                - padding: 转置卷积填充，默认 1。
                - bias: 是否使用偏置，默认 True。
                - activation: 激活函数类型，默认 'prelu'。
                - norm: 归一化类型，默认 None。
                """
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        """
                前向传播，执行转置卷积、归一化和激活操作。
                参数：
                - x: 输入张量，形状为 (batch_size, input_size, H, W)。
                返回：
                - out: 输出张量，形状为 (batch_size, output_size, H*stride, W*stride)。
                """
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class Decoder_MDCBlock1(torch.nn.Module):
    # ... (Decoder_MDCBlock1 代码保持不变) ...
    """
        Decoder_MDCBlock1: 多尺度解码器/融合块，用于融合不同尺度的特征。
        说明：
        - 支持多种融合模式（iter1, iter2, iter3, iter4），代码中主要使用 iter2。
        - 通过下采样和上采样操作，将高层特征与低层特征逐层融合，增强特征表达。
        - 用于解码器中，融合编码器跳跃连接的特征。
        """

    def __init__(self, num_filter, num_ft, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None,
                 mode='iter1'):
        """
                初始化 Decoder_MDCBlock1。
                参数：
                - num_filter: 基础通道数。
                - num_ft: 特征层级数量（低层特征数量 + 1）。
                - kernel_size: 卷积核大小，默认 4。
                - stride: 卷积步长，默认 2。
                - padding: 卷积填充，默认 1。
                - bias: 是否使用偏置，默认 True。
                - activation: 激活函数类型，默认 'prelu'。
                - norm: 归一化类型，默认 None。
                - mode: 融合模式（'iter1', 'iter2', 'iter3', 'iter4'），默认 'iter1'。
                """
        super(Decoder_MDCBlock1, self).__init__()
        self.mode = mode
        self.num_ft = num_ft - 1  # 特征层级的数量
        self.down_convs = nn.ModuleList()  # 下采样卷积列表
        self.up_convs = nn.ModuleList()  # 上采样反卷积列表
        # 根据层级数，创建对应的下采样和上采样卷积
        for i in range(self.num_ft):
            self.down_convs.append(
                ConvBlock(num_filter * (2 ** i), num_filter * (2 ** (i + 1)), kernel_size, stride, padding, bias,
                          activation, norm=None)
            )
            self.up_convs.append(
                DeconvBlock(num_filter * (2 ** (i + 1)), num_filter * (2 ** i), kernel_size, stride, padding, bias,
                            activation, norm=None)
            )

    def forward(self, ft_h, ft_l_list):
        """
                前向传播，融合高层特征和低层特征。
                参数：
                - ft_h: 高层特征张量。
                - ft_l_list: 低层特征张量列表。
                返回：
                - ft_fusion: 融合后的特征张量。
                """
        if self.mode == 'iter1' or self.mode == 'conv':
            # 模式1：
            ft_h_list = []
            for i in range(len(ft_l_list)):
                ft_h_list.append(ft_h)
                ft_h = self.down_convs[self.num_ft - len(ft_l_list) + i](ft_h)

            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft_fusion = self.up_convs[self.num_ft - i - 1](ft_fusion - ft_l_list[i]) + ft_h_list[
                    len(ft_l_list) - i - 1]

        if self.mode == 'iter2':
            # 模式2：(代码中使用的模式)
            # 核心思想：将当前特征 ft_h 与 ft_l_list 中的每个低层特征进行融合
            ft_fusion = ft_h  # 融合结果初始化为当前特征
            for i in range(len(ft_l_list)):  # 遍历所有低层特征
                ft = ft_fusion  #
                for j in range(self.num_ft - i):  # 1. 将当前融合特征下采样到与ft_l_list[i]相同的尺度
                    ft = self.down_convs[j](ft)

                ft = ft - ft_l_list[i]  # 2. 计算差异

                for j in range(self.num_ft - i):  # 3. 将差异上采样回原始尺度
                    ft = self.up_convs[self.num_ft - i - j - 1](ft)

                ft_fusion = ft_fusion + ft  # 4. 将差异（校正）加回到融合特征上

        if self.mode == 'iter3':
            # 模式3：
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft = ft_fusion
                for j in range(i + 1):
                    ft = self.down_convs[j](ft)
                ft = ft - ft_l_list[len(ft_l_list) - i - 1]
                for j in range(i + 1):
                    # print(j)
                    ft = self.up_convs[i + 1 - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter4':
            # 模式4：
            ft_fusion = ft_h
            for i in range(len(ft_l_list)):
                ft = ft_h
                for j in range(self.num_ft - i):
                    ft = self.down_convs[j](ft)
                ft = ft - ft_l_list[i]
                for j in range(self.num_ft - i):
                    ft = self.up_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        return ft_fusion


class make_dense(nn.Module):
    # ... (make_dense 代码保持不变) ...
    """
        make_dense: 密集连接块的单层实现，用于 RDB。
        说明：
        - 实现密集连接，通过卷积生成新特征并与输入特征拼接。
        - 用于残差密集块（RDB）中，增强特征的密集连接性。
        """

    def __init__(self, nChannels, growthRate, kernel_size=3):
        """
                初始化 make_dense 层。
                参数：
                - nChannels: 输入通道数。
                - growthRate: 输出通道数（增长率）。
                - kernel_size: 卷积核大小，默认 3。
        """
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)  # 将输入和输出在通道上拼接
        return out


class RDB(nn.Module):
    # ... (RDB 代码保持不变) ...
    """
        RDB: 残差密集块（Residual Dense Block）。
        说明：
        - 由多个 make_dense 层组成，堆叠形成密集连接网络。
        - 最后通过 1x1 卷积调整通道数，并添加残差连接。
        - 用于解码器中，增强特征的表达能力。
    """

    def __init__(self, nChannels, nDenselayer, growthRate, scale=1.0):
        """
                初始化 RDB。
                参数：
                - nChannels: 输入和输出通道数。
                - nDenselayer: make_dense 层的数量。
                - growthRate: 每个 make_dense 层的通道增长率。
                - scale: 残差缩放因子，默认 1.0。
        """
        super(RDB, self).__init__()
        nChannels_ = nChannels
        self.scale = scale
        modules = []
        for i in range(nDenselayer):  # 堆叠多个 'make_dense' 层
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)  # 1x1卷积，恢复通道数

    def forward(self, x):
        """
                前向传播，执行密集连接和残差连接。
                参数：
                - x: 输入张量，形状为 (batch_size, nChannels, H, W)。
                返回：
                - out: 输出张量，形状为 (batch_size, nChannels, H, W)。
                """
        out = self.dense_layers(x)
        out = self.conv_1x1(out) * self.scale  # 1x1卷积，并乘以一个缩放因子
        out = out + x  # 添加残差连接
        return out


class ConvLayer(nn.Module):
    # ... (ConvLayer 代码保持不变) ...
    """
        ConvLayer: 带反射填充的卷积层。
        说明：
        - 使用反射填充（ReflectionPad）来减少边界效应，适合图像处理任务。
        - 用于最终输出层或其他需要高质量特征的场景。
        """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        """
                初始化 ConvLayer。
                参数：
                - in_channels: 输入通道数。
                - out_channels: 输出通道数。
                - kernel_size: 卷积核大小。
                - stride: 卷积步长。
                """
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        """
                前向传播，执行反射填充和卷积。
                参数：
                - x: 输入张量，形状为 (batch_size, in_channels, H, W)。
                返回：
                - out: 输出张量，形状为 (batch_size, out_channels, H', W')。
                """
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(torch.nn.Module):
    # ... (UpsampleConvLayer 代码保持不变) ...
    """
        UpsampleConvLayer: 上采样层，使用最近邻插值加 1x1 卷积。
        说明：
        - 通过最近邻插值（nearest-exact）进行上采样，然后用 1x1 卷积调整通道数。
        - 用于解码器中，逐步恢复图像分辨率。
        """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        """
                初始化 UpsampleConvLayer。
                参数：
                - in_channels: 输入通道数。
                - out_channels: 输出通道数。
                - kernel_size: 用于计算插值后尺寸。
                - stride: 上采样倍数。
                """
        super(UpsampleConvLayer, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        """
                前向传播，执行上采样和 1x1 卷积。
                参数：
                - x: 输入张量，形状为 (batch_size, in_channels, H, W)。
                返回：
                - out: 输出张量，形状为 (batch_size, out_channels, H*stride, W*stride)。
        """
        h = (x.shape[2] - 1) * self.stride + self.kernel_size
        w = (x.shape[3] - 1) * self.stride + self.kernel_size
        x = F.interpolate(x, size=(h, w), mode="nearest-exact")  # 最近邻插值
        out = self.conv2d(x)  # 1x1 卷积
        return out


class ResidualBlock(torch.nn.Module):
    # ... (ResidualBlock 代码保持不变) ...
    """
        ResidualBlock: 标准残差块，包含两个卷积层和残差连接。
        说明：
        - 结构：Conv -> PReLU -> Conv -> 残差连接（带缩放因子 0.1）。
        - 用于解码器中的特征精炼。
        """

    def __init__(self, channels):
        """
                初始化 ResidualBlock。
                参数：
                - channels: 输入和输出通道数。
                """

        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.relu = nn.PReLU()

    def forward(self, x):
        """
                前向传播，执行残差连接。
                参数：
                - x: 输入张量，形状为 (batch_size, channels, H, W)。
                返回：
                - out: 输出张量，形状为 (batch_size, channels, H, W)。
                """
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1  # 第二个卷积的输出乘以0.1（残差缩放）
        out = torch.add(out, residual)  # 添加残差
        return out


# --- [新增] 通道注意力融合模块 ---
class ChannelAttentionFusion(nn.Module):
    # ... (ChannelAttentionFusion 代码保持不变) ...
    def __init__(self, in_channels, reduction=16, out_channels=None):
        super(ChannelAttentionFusion, self).__init__()
        self.in_channels = in_channels
        total_channels = 2 * in_channels  # 拼接后的总通道数
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(total_channels, total_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(total_channels // reduction, total_channels, bias=False),
            nn.Sigmoid()
        )

        # 可选的输出卷积层，用于调整最终输出通道数
        if out_channels is not None and out_channels != total_channels:
            self.output_conv = ConvLayer(total_channels, out_channels, kernel_size=1, stride=1)
        else:
            self.output_conv = None  # nn.Identity() PyTorch 1.6+

    def forward(self, x1, x2):
        # 1. 拼接特征
        fused = torch.cat((x1, x2), dim=1)  # (B, 2C, H, W)
        b, c, _, _ = fused.size()

        # 2. 计算通道注意力权重
        y = self.avg_pool(fused).view(b, c)  # (B, 2C)
        y = self.fc(y).view(b, c, 1, 1)  # (B, 2C, 1, 1)

        # 3. 应用注意力权重
        attended_features = fused * y.expand_as(fused)  # (B, 2C, H, W)

        # 4. (可选) 调整输出通道
        if self.output_conv is not None:
            output = self.output_conv(attended_features)
        else:
            output = attended_features

        return output


# --- [新增结束] ---


# --- [区域补全范式] VIFNetInconsistencyTeacher 模型 ---
class VIFNetInconsistencyTeacher(nn.Module):
    """
    Region-completion paradigm teacher model. HAPM mask splits into two paths:
      - Pass 1 (lightweight dual-stream): pure IR structure + per-pixel fusion weight beta
      - Pass 2 (Res2Net): M=1 -> pure IR fill, M=0 -> asymmetric IR residual fusion
      - Color restoration: in-image Cross-Attention, K/V from M=0 reliable regions only
    """

    def __init__(
        self,
        res_blocks=18,
        semantic_dim=128,
        num_color_prototypes=32,
        transport_temperature=0.07,
        fusion_temperature=0.07,
        verify_threshold=0.2,
        verify_temperature=0.1,
    ):
        super(VIFNetInconsistencyTeacher, self).__init__()

        # Legacy CMDN is intentionally disabled in active TMM.
        # The active Teacher path uses HDE + mask_head + GumbelSigmoidBinarizer.
        # Keep this attribute only for old checkpoints and introspection; forward
        # does not call CMDN/Otsu/sky-mask routing.
        self.cmdn = None

        self.hde = HDE()
        self.gumbel_binarizer = GumbelSigmoidBinarizer(tau=1.0, hard=True, threshold=0.5)
        self.mask_head = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        self.fusion_weight_heads = nn.ModuleList([
            nn.Sequential(nn.Conv2d(2049, 1, kernel_size=3, padding=1), nn.Sigmoid()),  # H/16
            nn.Sequential(nn.Conv2d(1025, 1, kernel_size=3, padding=1), nn.Sigmoid()),  # H/8
            nn.Sequential(nn.Conv2d(513, 1, kernel_size=3, padding=1), nn.Sigmoid()),   # H/4
            nn.Sequential(nn.Conv2d(129, 1, kernel_size=3, padding=1), nn.Sigmoid()),   # H/2
        ])


        # --- [新增] 阶段一 (Pass 1) 模块 (来自代码库 B) ---
        # VIFnet (代码库 B) 默认 n_feat=64
        b_n_feat = 64
        b_kernel_size = 3
        b_bias = False

        # Pass 1 VIS 流 (轻量级)
        self.vis_layer1_b = nn.Sequential(Conv_B(3, b_n_feat, b_kernel_size, bias=b_bias),
                                          CPAB(b_n_feat, b_kernel_size, b_bias),
                                          CPAB(b_n_feat, b_kernel_size, b_bias))
        self.encoder_b_vis = Encoder_B(b_n_feat, b_kernel_size, b_bias, atten=False)
        self.decoder_b_vis = Decoder_B(b_n_feat, b_kernel_size, b_bias, residual=True)
        self.dsfe_vis = DSFE(b_n_feat, b_kernel_size, b_bias)

        # Pass 1 IR 流 (轻量级)
        self.ir_layer1_b = nn.Sequential(Conv_B(3, b_n_feat, b_kernel_size, bias=b_bias),
                                         CPAB(b_n_feat, b_kernel_size, b_bias),
                                         CPAB(b_n_feat, b_kernel_size, b_bias))
        self.encoder_b_ir = Encoder_B(b_n_feat, b_kernel_size, b_bias, atten=False)
        self.decoder_b_ir = Decoder_B(b_n_feat, b_kernel_size, b_bias, residual=True)
        self.dsfe_ir = DSFE(b_n_feat, b_kernel_size, b_bias)
        # --- [新增结束] ---

        # --- 阶段二 (Pass 2) 模块 (来自代码库 A) ---

        # --- 可见光流 (主网络) ---
        # [修改]：Res2Net 现在内部包含了 inject_conv 模块
        self.encoder_vis = Res2Net(Bottle2neck, [3, 4, 23], baseWidth=26, scale=4, in_channels=3)
        # ... (加载权重代码保留) ...
        try:
            res2net101_full = Pre_Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4)
            # [请注意]：请确保你本地 'D:/...' 路径下存在此文件
            pretrained_path = '/root/CoA-main_v10_Sup3_canny/model/imagenet_model/res2net101_v1b_26w_4s-0812c246.pth'
            if not os.path.exists(pretrained_path):
                raise FileNotFoundError(f"预训练权重文件未找到: {pretrained_path}")
            res2net101_full.load_state_dict(torch.load(pretrained_path, map_location='cpu'), strict=False)
            pretrained_dict = res2net101_full.state_dict()
            model_dict = self.encoder_vis.state_dict()
            key_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(key_dict)
            self.encoder_vis.load_state_dict(model_dict)
            print("Successfully loaded pretrained weights for visible stream encoder.")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights for visible stream encoder. {e}")

        # ... (A 库的 CRA_vis, H_vis, dehaze_vis, decoder_vis 模块定义保持不变) ...
        self.CRA1_vis = nn.Conv2d(1024, 256, kernel_size=1)
        self.CRA2_vis = nn.Conv2d(512, 128, kernel_size=1)
        self.CRA3_vis = nn.Conv2d(256, 64, kernel_size=1)
        self.CRA4_vis = nn.Conv2d(64, 32, kernel_size=1)
        self.H1_vis = nn.Conv2d(256, 128, kernel_size=1)
        self.H2_vis = nn.Conv2d(128, 64, kernel_size=1)
        self.H3_vis = nn.Conv2d(64, 32, kernel_size=1)
        self.H4_vis = nn.Conv2d(32, 16, kernel_size=1)
        self.dehaze_vis = nn.Sequential(*[ResidualBlock(256) for _ in range(res_blocks)])
        self.convd16x_vis = UpsampleConvLayer(256, 128, kernel_size=3, stride=2)
        self.dense_4_vis = nn.Sequential(ResidualBlock(128), ResidualBlock(128), ResidualBlock(128))
        self.conv_4_vis = RDB(64, 4, 64)
        self.fusion_4_vis = Decoder_MDCBlock1(64, 2, mode='iter2')
        self.convd8x_vis = UpsampleConvLayer(128, 64, kernel_size=3, stride=2)
        self.dense_3_vis = nn.Sequential(ResidualBlock(64), ResidualBlock(64), ResidualBlock(64))
        self.conv_3_vis = RDB(32, 4, 32)
        self.fusion_3_vis = Decoder_MDCBlock1(32, 3, mode='iter2')
        self.convd4x_vis = UpsampleConvLayer(64, 32, kernel_size=3, stride=2)
        self.dense_2_vis = nn.Sequential(ResidualBlock(32), ResidualBlock(32), ResidualBlock(32))
        self.conv_2_vis = RDB(16, 4, 16)
        self.fusion_2_vis = Decoder_MDCBlock1(16, 4, mode='iter2')
        self.convd2x_vis = UpsampleConvLayer(32, 16, kernel_size=3, stride=2)
        self.dense_1_vis = nn.Sequential(ResidualBlock(16), ResidualBlock(16), ResidualBlock(16))
        self.conv_1_vis = RDB(8, 4, 8)
        self.fusion_1_vis = Decoder_MDCBlock1(8, 5, mode='iter2')

        # --- [修改] 保留 IR 流解码器 (用于跨模态损失) ---
        self.encoder_ir = Res2Net(Bottle2neck, [3, 4, 23], baseWidth=26, scale=4, in_channels=3)
        # ... (加载权重代码保留) ...
        try:
            # (与可见光流加载方式相同)
            res2net101_full_ir = Pre_Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4)
            # [请注意]：请确保你本地 'D:/...' 路径下存在此文件
            pretrained_path_ir = '/root/CoA-main_v10_Sup3_canny/model/imagenet_model/res2net101_v1b_26w_4s-0812c246.pth'
            if not os.path.exists(pretrained_path_ir):
                raise FileNotFoundError(f"预训练权重文件未找到: {pretrained_path_ir}")
            res2net101_full_ir.load_state_dict(torch.load(pretrained_path_ir, map_location='cpu'), strict=False)
            pretrained_dict_ir = res2net101_full_ir.state_dict()
            model_dict_ir = self.encoder_ir.state_dict()
            key_dict_ir = {k: v for k, v in pretrained_dict_ir.items() if
                           k in model_dict_ir and model_dict_ir[k].shape == v.shape}
            model_dict_ir.update(key_dict_ir)
            self.encoder_ir.load_state_dict(model_dict_ir)
            print("Successfully loaded pretrained weights for infrared stream encoder.")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights for infrared stream encoder. {e}")

        self.CRA1_ir = nn.Conv2d(1024, 256, kernel_size=1)
        self.CRA2_ir = nn.Conv2d(512, 128, kernel_size=1)
        self.CRA3_ir = nn.Conv2d(256, 64, kernel_size=1)
        self.CRA4_ir = nn.Conv2d(64, 32, kernel_size=1)
        self.dehaze_ir = nn.Sequential(*[ResidualBlock(256) for _ in range(res_blocks)])
        self.convd16x_ir = UpsampleConvLayer(256, 128, kernel_size=3, stride=2)
        self.dense_4_ir = nn.Sequential(ResidualBlock(128), ResidualBlock(128), ResidualBlock(128))
        self.conv_4_ir = RDB(64, 4, 64)
        self.fusion_4_ir = Decoder_MDCBlock1(64, 2, mode='iter2')
        self.convd8x_ir = UpsampleConvLayer(128, 64, kernel_size=3, stride=2)
        self.dense_3_ir = nn.Sequential(ResidualBlock(64), ResidualBlock(64), ResidualBlock(64))
        self.conv_3_ir = RDB(32, 4, 32)
        self.fusion_3_ir = Decoder_MDCBlock1(32, 3, mode='iter2')
        self.convd4x_ir = UpsampleConvLayer(64, 32, kernel_size=3, stride=2)
        self.dense_2_ir = nn.Sequential(ResidualBlock(32), ResidualBlock(32), ResidualBlock(32))
        self.conv_2_ir = RDB(16, 4, 16)
        self.fusion_2_ir = Decoder_MDCBlock1(16, 4, mode='iter2')
        self.convd2x_ir = UpsampleConvLayer(32, 16, kernel_size=3, stride=2)
        self.dense_1_ir = nn.Sequential(ResidualBlock(16), ResidualBlock(16), ResidualBlock(16))
        self.conv_1_ir = RDB(8, 4, 8)
        self.fusion_1_ir = Decoder_MDCBlock1(8, 5, mode='iter2')
        # --- [保留 IR 流结束] ---

        # --- [修改] 移除 final_fusion 并调整 conv_output ---
        # self.final_fusion = ChannelAttentionFusion(in_channels=16, reduction=4, out_channels=32)

        # [已移除] CLIP 视觉编码器加载 — 颜色恢复改为纯图内 Cross-Attention

        # CLIP 输入归一化参数（ViT-B/32 标准值，HDE 的 x_vis_01 反归一化必需）
        self.register_buffer(
            'clip_input_mean',
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'clip_input_std',
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        )

        self.semantic_dim = semantic_dim
        self.num_color_prototypes = num_color_prototypes
        self.transport_temperature = transport_temperature
        self.fusion_temperature = fusion_temperature
        self.verify_threshold = verify_threshold
        self.verify_temperature = verify_temperature
        self.shared_semantic_proj = SharedSemanticProjection(256, semantic_dim)
        self.color_transport = CrossModalSemanticColorTransport(
            in_channels=256,
            semantic_dim=semantic_dim,
            num_prototypes=num_color_prototypes,
            temperature=transport_temperature,
            shared_proj=self.shared_semantic_proj,
        )
        self.bidir_fusion = BiDirectionalSemanticFusion(
            in_channels=256,
            semantic_dim=semantic_dim,
            temperature=fusion_temperature,
            verify_threshold=verify_threshold,
            verify_temperature=verify_temperature,
            shared_proj=self.shared_semantic_proj,
        )

        # [修改] conv_output 现在直接接收来自 vis_features 的 16 个通道
        self.conv_output = ConvLayer(16, 3, kernel_size=3, stride=1)
        # --- [修改结束] ---

    # --- [删除] _process_vis_decoder 和 _process_ir_stream ---
    # (这两个函数的功能将被内联并重构到新的 forward 方法中)

    def set_gumbel_tau(self, tau):
        self.gumbel_binarizer.set_tau(tau)

    # Debug/ablation helper only. Formal training and formal inference should
    # pass no external override so binary_mask comes from internal HDE + Gumbel.
    def _override_binary_mask(self, binary_mask, override_mask):
        override_mask = override_mask.to(device=binary_mask.device, dtype=binary_mask.dtype)
        if override_mask.dim() == 3:
            override_mask = override_mask.unsqueeze(1)
        override_mask = F.interpolate(override_mask, size=binary_mask.shape[2:], mode='nearest')
        return (override_mask >= 0.5).float()

    # --- [重写] forward 方法：合成域 HDE + Gumbel 区域补全主链路 ---
    def forward(self, x_vis, x_ir, haze_mask=None, return_dict=False, debug_force_mask=None, **kwargs):
        # haze_mask/debug_force_mask are retained strictly for diagnostics and
        # ablations. Formal Eval_EMA inference must leave them as None, using
        # the model-internal HDE + mask_head + Gumbel binary_mask below.
        x_vis_01 = (x_vis * self.clip_input_std + self.clip_input_mean).clamp(0.0, 1.0)
        density_map, density_feat = self.hde(x_vis_01, return_feat=True)
        mask_logits = self.mask_head(density_feat)
        mask_prob, binary_mask = self.gumbel_binarizer(mask_logits, is_logits=True)

        if haze_mask is not None:
            binary_mask = self._override_binary_mask(binary_mask, haze_mask)
        if debug_force_mask is not None:
            binary_mask = self._override_binary_mask(binary_mask, debug_force_mask)

        ir_content_outputs, _ = self.encoder_ir(x_ir)

        fused_outputs, original_outputs, region_debug = self.encoder_vis(
            x_vis,
            ir_feat_list=ir_content_outputs,
            haze_mask=binary_mask,
            density_map=density_map,
            fusion_weight_heads=self.fusion_weight_heads,
            bidir_fusion=self.bidir_fusion,
            return_region_debug=True,
        )

        x_layer3_fused, x_layer2_fused, x_layer1_fused, x_layer0_safe = fused_outputs

        # Current region_debug order is [H/16, H/8, H/4, H/2].
        # If that order changes, update this H/4 semantic color transport hook.
        ir_h4 = region_debug["ir_feats"][2]
        vis_h4 = region_debug["vis_feats"][2]
        assert ir_h4.shape[1] == 256, f"Expected H/4 IR feat 256ch, got {ir_h4.shape}"
        assert vis_h4.shape[1] == 256, f"Expected H/4 VIS feat 256ch, got {vis_h4.shape}"
        assert ir_h4.shape[2:] == vis_h4.shape[2:], "IR/VIS H4 feature spatial size mismatch"

        res16x_vis = self.CRA1_vis(x_layer3_fused)
        res8x_vis = self.CRA2_vis(x_layer2_fused)
        res4x_vis = self.CRA3_vis(x_layer1_fused)
        res2x_vis = self.CRA4_vis(x_layer0_safe)

        in_ft = res16x_vis
        res16x_dehazed = self.dehaze_vis(in_ft) + res16x_vis
        res16x_1, res16x_2 = res16x_dehazed.split(
            [(res16x_dehazed.size(1) // 2), (res16x_dehazed.size(1) // 2)], dim=1
        )
        feature_mem_up = [res16x_1]

        res16x_up = self.convd16x_vis(res16x_dehazed)
        res16x_up = F.interpolate(res16x_up, size=res8x_vis.size()[2:], mode='bilinear', align_corners=False)
        res8x_fused = torch.add(res16x_up, res8x_vis)
        res8x_dense = self.dense_4_vis(res8x_fused) + res8x_fused
        res8x_1, res8x_2 = res8x_dense.split([(res8x_dense.size(1) // 2), (res8x_dense.size(1) // 2)], dim=1)
        res8x_1 = self.fusion_4_vis(res8x_1, feature_mem_up)
        res8x_2 = self.conv_4_vis(res8x_2)
        feature_mem_up.append(res8x_1)
        res8x_out = torch.cat((res8x_1, res8x_2), dim=1)

        res8x_up = self.convd8x_vis(res8x_out)
        res8x_up = F.interpolate(res8x_up, size=res4x_vis.size()[2:], mode='bilinear', align_corners=False)
        res4x_fused = torch.add(res8x_up, res4x_vis)
        res4x_dense = self.dense_3_vis(res4x_fused) + res4x_fused
        res4x_1, res4x_2 = res4x_dense.split([(res4x_dense.size(1) // 2), (res4x_dense.size(1) // 2)], dim=1)
        res4x_1 = self.fusion_3_vis(res4x_1, feature_mem_up)
        res4x_2 = self.conv_3_vis(res4x_2)
        feature_mem_up.append(res4x_1)
        res4x_out = torch.cat((res4x_1, res4x_2), dim=1)

        res4x_up = self.convd4x_vis(res4x_out)
        res4x_up = F.interpolate(res4x_up, size=res2x_vis.size()[2:], mode='bilinear', align_corners=False)
        res2x_fused = torch.add(res4x_up, res2x_vis)
        res2x_dense = self.dense_2_vis(res2x_fused) + res2x_fused
        res2x_1, res2x_2 = res2x_dense.split([(res2x_dense.size(1) // 2), (res2x_dense.size(1) // 2)], dim=1)
        res2x_1 = self.fusion_2_vis(res2x_1, feature_mem_up)
        res2x_2 = self.conv_2_vis(res2x_2)
        feature_mem_up.append(res2x_1)
        res2x_out = torch.cat((res2x_1, res2x_2), dim=1)

        res2x_up = self.convd2x_vis(res2x_out)
        res2x_up = F.interpolate(res2x_up, size=x_vis.size()[2:], mode='bilinear', align_corners=False)
        x_fused = res2x_up
        x_dense = self.dense_1_vis(x_fused) + x_fused
        x_1, x_2 = x_dense.split([(x_dense.size(1) // 2), (x_dense.size(1) // 2)], dim=1)
        x_1 = self.fusion_1_vis(x_1, feature_mem_up)
        x_2 = self.conv_1_vis(x_2)
        vis_features = torch.cat((x_1, x_2), dim=1)

        pred_raw = torch.sigmoid(self.conv_output(vis_features)).clamp(0.0, 1.0)
        color_out = self.color_transport(
            ir_feat=ir_h4,
            vis_feat=vis_h4,
            x_vis_01=x_vis_01,
            haze_mask=binary_mask,
        )
        transported_rgb = color_out["transported_rgb"].clamp(0.0, 1.0)
        M_full = F.interpolate(binary_mask, size=pred_raw.shape[-2:], mode="nearest")
        pred_clear = (M_full * transported_rgb + (1.0 - M_full) * pred_raw).clamp(0.0, 1.0)

        if return_dict:
            return_dict = {
                "pred_clear": pred_clear,
                "pred_raw": pred_raw,
                "density_map": density_map,
                "mask_logits": mask_logits,
                "mask_prob": mask_prob,
                "binary_mask": binary_mask,
                "fused_feats": region_debug["fused_feats"],
                "ir_feats": region_debug["ir_feats"],
                "vis_feats": region_debug["vis_feats"],
                "fusion_weights": region_debug["fusion_weights"],
                "fusion_debug_h4": region_debug.get("fusion_debug_h4"),
            }
            return_dict.update(color_out)
            return return_dict

        # Compatibility tuple: item 0 remains pred_clear. Item 4 is the new
        # HDE/Gumbel binary_mask and no longer has old CMDN m_hard semantics.
        return (
            pred_clear,
            None,
            None,
            None,
            binary_mask,
            None,
            None,
            None,
            None,
            None,
            None,
        )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = VIFNetInconsistencyTeacher(semantic_dim=128, num_color_prototypes=32, transport_temperature=0.07).to(device)
    dummy_input_vis = torch.randn(1, 3, 256, 256).to(device)
    dummy_input_ir = torch.randn(1, 3, 256, 256).to(device)
    debug_mask = torch.zeros(1, 1, 256, 256, device=device)
    debug_mask[:, :, :, :128] = 1.0

    with torch.no_grad():
        # debug_force_mask is only for smoke tests/ablations; formal inference
        # uses the model-internal HDE + Gumbel binary_mask.
        out = net(dummy_input_vis, dummy_input_ir, return_dict=True, debug_force_mask=debug_mask)

    print("pred_clear:", out["pred_clear"].shape)
    print("pred_raw:", out["pred_raw"].shape)
    print("transported_rgb:", out["transported_rgb"].shape)
    print("transported_rgb_feat:", out["transported_rgb_feat"].shape)
    print("semantic_ir:", out["semantic_ir"].shape)
    print("semantic_vis:", out["semantic_vis"].shape)
    print("proto_keys:", out["proto_keys"].shape)
    print("proto_values:", out["proto_values"].shape)
    print("proto_attn:", out["proto_attn"].shape)
    print("proto_assign:", out["proto_assign"].shape)
    print("max_sim_map:", out["max_sim_map"].shape)
    print("density_map:", out["density_map"].shape)
    print("mask_logits:", out["mask_logits"].shape)
    print("mask_prob:", out["mask_prob"].shape)
    print("binary_mask:", out["binary_mask"].shape)
    print("fused_feats:", [x.shape for x in out["fused_feats"]])
    print("ir_feats:", [x.shape for x in out["ir_feats"]])
    print("vis_feats:", [x.shape for x in out["vis_feats"]])
    print("fusion_weights:", [x.shape for x in out["fusion_weights"]])
    assert out["pred_clear"].shape == out["pred_raw"].shape == out["transported_rgb"].shape == (1, 3, 256, 256)
    assert out["semantic_ir"].shape[1] == net.semantic_dim
    assert out["semantic_vis"].shape[1] == net.semantic_dim
    assert out["semantic_ir"].shape[2:] == out["semantic_vis"].shape[2:]
    Hf, Wf = out["semantic_ir"].shape[2:]
    assert out["proto_attn"].shape == (1, Hf * Wf, net.num_color_prototypes)
    assert torch.isfinite(out["proto_attn"]).all()
    M_full = F.interpolate(out["binary_mask"], size=out["pred_raw"].shape[-2:], mode="nearest")
    assert torch.allclose(
        out["pred_clear"][M_full.expand_as(out["pred_clear"]) == 1],
        out["transported_rgb"][M_full.expand_as(out["transported_rgb"]) == 1],
        atol=1e-6,
    )
    assert torch.allclose(
        out["pred_clear"][M_full.expand_as(out["pred_clear"]) == 0],
        out["pred_raw"][M_full.expand_as(out["pred_raw"]) == 0],
        atol=1e-6,
    )

    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total_params:", pytorch_total_params)
