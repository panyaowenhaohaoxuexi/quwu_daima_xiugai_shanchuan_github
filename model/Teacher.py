# -*- coding: utf-8 -*-
"""
*** 修改后的模型： VIFNetInconsistencyTeacher with Channel Attention Fusion ***
- 保留 DualStreamTeacher 的双流结构和特征提取器。
- 引入 VIFNet 的不一致性计算和加权融合思想（注入可见光流解码器）。
- *** 修改了最终融合方式：使用通道注意力机制融合 vis_features 和 ir_features，替代简单的 torch.cat ***
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import torchvision.transforms.functional as TF
from collections import OrderedDict

# --- VIFNet 不一致性函数 f(x, y) (保持不变) ---
def f(x, y):
    """VIFNet inconsistency function (code version)"""
    return (1 - x) * (1 - y) + 1 / 2 * x * y

# --- 基础模块 (SobelEdgeDetector, Pre_Res2Net, Bottle2neck, Res2Net(3通道输入), ConvBlock, DeconvBlock, Decoder_MDCBlock1, make_dense, RDB, ConvLayer, UpsampleConvLayer, ResidualBlock) ---
# ... (这些基础模块的代码与上一个版本相同，确保 Res2Net 输入为 3 通道，这里省略以保持简洁) ...
class SobelEdgeDetector(nn.Module):
    def __init__(self):
        super(SobelEdgeDetector, self).__init__()
        sobel_kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        sobel_kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        self.kernel_x = sobel_kernel_x.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
        self.kernel_y = sobel_kernel_y.view(1, 1, 3, 3).repeat(1, 1, 1, 1)
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_x.weight = nn.Parameter(self.kernel_x, requires_grad=False)
        self.conv_y.weight = nn.Parameter(self.kernel_y, requires_grad=False)

    def forward(self, x):
        if x.shape[1] != 1:
            if x.shape[1] == 3:
                 x = TF.rgb_to_grayscale(x)
            else:
                 print("警告: Sobel 输入不是单通道或三通道，将只使用第一个通道。")
                 x = x[:, 0:1, :, :]
        grad_x = self.conv_x(x)
        grad_y = self.conv_y(x)
        edge_magnitude = torch.abs(grad_x) + torch.abs(grad_y)
        return edge_magnitude


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
            nn.Conv2d(3, 32, 3, 2, 1, bias=False), # *** 注意这里的输入通道是 3 ***
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
        *** 确保输入为 3 通道 ***
    """
    def __init__(self, block, layers, baseWidth=26, scale=4, in_channels=3): # 添加 in_channels 参数
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        # 使用 in_channels 参数
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False), # *** 使用 in_channels ***
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

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        """
                构建 Res2Net 的一个阶段。
                参数：
                - block: Bottle2neck 类。
                - planes: 基础通道数。
                - blocks: 该阶段的块数量。
                - stride: 第一个块的步长。
                返回：
                - nn.Sequential: 包含所有 Bottle2neck 块的序列。
        """
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

    def forward(self, x): # 输入 x 应该是 3 通道

        x = self.conv1(x) # 第一层处理 3 通道输入
        x = self.bn1(x)
        x = self.relu(x)

        x_layer0 = x  # 保存conv1后的特征 (64 通道)
        x = self.maxpool(x)

        x_layer1 = self.layer1(x)  # layer1 输出
        x_layer2 = self.layer2(x_layer1)  # layer2 输出
        x_layer3 = self.layer3(x_layer2)  # layer3 输出

        return x_layer3, x_layer2, x_layer1, x_layer0

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

    def __init__(self, in_channels, reduction=16, out_channels=None):
        super(ChannelAttentionFusion, self).__init__()
        self.in_channels = in_channels
        total_channels = 2 * in_channels # 拼接后的总通道数
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
            self.output_conv = None # nn.Identity() PyTorch 1.6+

    def forward(self, x1, x2):
        # 1. 拼接特征
        fused = torch.cat((x1, x2), dim=1) # (B, 2C, H, W)
        b, c, _, _ = fused.size()

        # 2. 计算通道注意力权重
        y = self.avg_pool(fused).view(b, c) # (B, 2C)
        y = self.fc(y).view(b, c, 1, 1) # (B, 2C, 1, 1)

        # 3. 应用注意力权重
        attended_features = fused * y.expand_as(fused) # (B, 2C, H, W)

        # 4. (可选) 调整输出通道
        if self.output_conv is not None:
             output = self.output_conv(attended_features)
        else:
             output = attended_features

        return output
# --- [新增结束] ---


# --- 修改后的 VIFNetInconsistencyTeacher 模型 ---
class VIFNetInconsistencyTeacher(nn.Module):
    """
    双流教师模型，在可见光流中融入了基于 VIFNet 不一致性加权的红外特征。
    *** 修改了最终融合方式：使用通道注意力 ***
    """
    def __init__(self, res_blocks=18):
        super(VIFNetInconsistencyTeacher, self).__init__()

        # --- 可见光流 (保持不变) ---
        self.encoder_vis = Res2Net(Bottle2neck, [3, 4, 23], baseWidth=26, scale=4, in_channels=3)
        # 加载预训练权重 (省略代码)
        try:
            res2net101_full = Pre_Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4)
            # --- [修改] ---
            # 修改预训练权重路径为你本地的路径
            pretrained_path = 'D:/liu_lan_qi_xia_zai/CoA-main_daima_xiugai_jiehe_ir_edge_xiugai_teacher_v5/model/imagenet_model/res2net101_v1b_26w_4s-0812c246.pth'
            # --- [修改结束] ---
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

        # --- 红外流 (保持不变) ---
        self.encoder_ir = Res2Net(Bottle2neck, [3, 4, 23], baseWidth=26, scale=4, in_channels=3)
        # 加载预训练权重 (省略代码)
        try:
            # (与可见光流加载方式相同)
            res2net101_full_ir = Pre_Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4)
            # --- [修改] ---
            # 修改预训练权重路径为你本地的路径
            pretrained_path_ir = 'D:/liu_lan_qi_xia_zai/CoA-main_daima_xiugai_jiehe_ir_edge_xiugai_teacher_v5/model/imagenet_model/res2net101_v1b_26w_4s-0812c246.pth'
            # --- [修改结束] ---
            if not os.path.exists(pretrained_path_ir):
                 raise FileNotFoundError(f"预训练权重文件未找到: {pretrained_path_ir}")
            res2net101_full_ir.load_state_dict(torch.load(pretrained_path_ir, map_location='cpu'), strict=False)
            pretrained_dict_ir = res2net101_full_ir.state_dict()
            model_dict_ir = self.encoder_ir.state_dict()
            key_dict_ir = {k: v for k, v in pretrained_dict_ir.items() if k in model_dict_ir and model_dict_ir[k].shape == v.shape}
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

        # --- [修改] 最终融合与输出 ---
        # 1. 通道注意力融合模块
        # 输入是 vis_features (16) 和 ir_features (16)
        self.final_fusion = ChannelAttentionFusion(in_channels=16, reduction=4, out_channels=32) # 输出保持32通道
        # 2. 最终输出卷积层 (输入通道与 final_fusion 输出匹配)
        self.conv_output = ConvLayer(16, 3, kernel_size=3, stride=1) # <--- 已修复
        # --- [修改结束] ---

        # (VIFNet 加权融合所需的 1x1 卷积层保持不变)
        # self.proj16x = nn.Conv2d(256, 256, 1)
        # self.proj8x = nn.Conv2d(128, 128, 1)
        # self.proj4x = nn.Conv2d(64, 64, 1)
        # self.proj2x = nn.Conv2d(32, 32, 1)
        # --- [修改2：新增 编码器端的注意力融合模块实例] ---
        # 对应 CRA 层的输出通道 (256, 128, 64, 32)
        self.enc_fusion_16x = AttentionFusionBlock(channel=256)
        self.enc_fusion_8x = AttentionFusionBlock(channel=128)
        self.enc_fusion_4x = AttentionFusionBlock(channel=64)
        self.enc_fusion_2x = AttentionFusionBlock(channel=32)
        # --- [修改2：结束] ---

    # --- _process_vis_stream_with_fusion 和 _process_ir_stream 保持不变 ---
    # ... (这两个函数的代码与上一个版本相同，这里省略) ...
        # --- [修改4：重构可见光解码器] ---
        # (此函数取代了 _process_vis_stream_with_fusion)
    def _process_vis_decoder(self,
                                 fused_vis_cra_list,  # [新] 接收融合后的CRA特征 [res16x_vis, res8x_vis, ...]
                                 inf_weights,  # [新] 接收不一致性权重 [inf_weight_16x, ...]
                                 dehaze,
                                 convd16x, dense_4, conv_4, fusion_4,
                                 convd8x, dense_3, conv_3, fusion_3,
                                 convd4x, dense_2, conv_2, fusion_2,
                                 convd2x, dense_1, conv_1, fusion_1,
                                 target_size):  # [新] 接收最终输出的目标尺寸
            """
            [修改后的] 可见光解码器处理函数。
            - 直接使用传入的融合特征 (fused_vis_cra_list)。
            - 在跳跃连接前融合不一致性权重 (inf_weights)。
            - 不再计算 H 特征（移至 forward 方法中）。
            """
            # --- 1. 解包早期融合的CRA特征 ---
            res16x_vis, res8x_vis, res4x_vis, res2x_vis = fused_vis_cra_list

            # --- 2. 解包不一致性权重 ---
            inf_weight_16x, inf_weight_8x, inf_weight_4x, inf_weight_2x = inf_weights

            # --- 3. 融合不一致性特征 ---
            # (确保尺寸一致)
            inf_weight_16x = F.interpolate(inf_weight_16x, size=res16x_vis.shape[2:], mode='bilinear',
                                           align_corners=False) if inf_weight_16x.shape[2:] != res16x_vis.shape[
                                                                                               2:] else inf_weight_16x
            inf_weight_8x = F.interpolate(inf_weight_8x, size=res8x_vis.shape[2:], mode='bilinear',
                                          align_corners=False) if inf_weight_8x.shape[2:] != res8x_vis.shape[
                                                                                             2:] else inf_weight_8x
            inf_weight_4x = F.interpolate(inf_weight_4x, size=res4x_vis.shape[2:], mode='bilinear',
                                          align_corners=False) if inf_weight_4x.shape[2:] != res4x_vis.shape[
                                                                                             2:] else inf_weight_4x
            inf_weight_2x = F.interpolate(inf_weight_2x, size=res2x_vis.shape[2:], mode='bilinear',
                                          align_corners=False) if inf_weight_2x.shape[2:] != res2x_vis.shape[
                                                                                             2:] else inf_weight_2x

            # (早期融合特征 + 不一致性权重)
            res16x_vis_fused = res16x_vis + inf_weight_16x
            res8x_vis_fused = res8x_vis + inf_weight_8x
            res4x_vis_fused = res4x_vis + inf_weight_4x
            res2x_vis_fused = res2x_vis + inf_weight_2x
            # --- [修改4：结束] ---

            # Dehaze (使用融合后的 res16x)
            in_ft = res16x_vis_fused  # 使用融合后的特征
            res16x_dehazed = dehaze(in_ft) + res16x_vis_fused  # 残差连接也用融合后的

            res16x_1, res16x_2 = res16x_dehazed.split([(res16x_dehazed.size(1) // 2), (res16x_dehazed.size(1) // 2)],
                                                      dim=1)
            feature_mem_up = [res16x_1]

            # Decoder Stage 1 (使用融合后的 res8x_vis_fused)
            res16x_up = convd16x(res16x_dehazed)
            res16x_up = F.interpolate(res16x_up, size=res8x_vis_fused.size()[2:], mode='bilinear', align_corners=False)
            res8x_fused = torch.add(res16x_up, res8x_vis_fused)  # <--- 使用融合后的 res8x
            res8x_dense = dense_4(res8x_fused) + res8x_fused
            res8x_1, res8x_2 = res8x_dense.split([(res8x_dense.size(1) // 2), (res8x_dense.size(1) // 2)], dim=1)
            res8x_1 = fusion_4(res8x_1, feature_mem_up)
            res8x_2 = conv_4(res8x_2)
            feature_mem_up.append(res8x_1)
            res8x_out = torch.cat((res8x_1, res8x_2), dim=1)

            # Decoder Stage 2 (使用融合后的 res4x_vis_fused)
            res8x_up = convd8x(res8x_out)
            res8x_up = F.interpolate(res8x_up, size=res4x_vis_fused.size()[2:], mode='bilinear', align_corners=False)
            res4x_fused = torch.add(res8x_up, res4x_vis_fused)  # <--- 使用融合后的 res4x
            res4x_dense = dense_3(res4x_fused) + res4x_fused
            res4x_1, res4x_2 = res4x_dense.split([(res4x_dense.size(1) // 2), (res4x_dense.size(1) // 2)], dim=1)
            res4x_1 = fusion_3(res4x_1, feature_mem_up)
            res4x_2 = conv_3(res4x_2)
            feature_mem_up.append(res4x_1)
            res4x_out = torch.cat((res4x_1, res4x_2), dim=1)

            # Decoder Stage 3 (使用融合后的 res2x_vis_fused)
            res4x_up = convd4x(res4x_out)
            res4x_up = F.interpolate(res4x_up, size=res2x_vis_fused.size()[2:], mode='bilinear', align_corners=False)
            res2x_fused = torch.add(res4x_up, res2x_vis_fused)  # <--- 使用融合后的 res2x
            res2x_dense = dense_2(res2x_fused) + res2x_fused
            res2x_1, res2x_2 = res2x_dense.split([(res2x_dense.size(1) // 2), (res2x_dense.size(1) // 2)], dim=1)
            res2x_1 = fusion_2(res2x_1, feature_mem_up)
            res2x_2 = conv_2(res2x_2)
            feature_mem_up.append(res2x_1)
            res2x_out = torch.cat((res2x_1, res2x_2), dim=1)

            # Decoder Stage 4
            res2x_up = convd2x(res2x_out)
            # [修改4] 使用传入的 target_size
            res2x_up = F.interpolate(res2x_up, size=target_size, mode='bilinear', align_corners=False)
            x_fused = res2x_up
            x_dense = dense_1(x_fused) + x_fused
            x_1, x_2 = x_dense.split([(x_dense.size(1) // 2), (x_dense.size(1) // 2)], dim=1)
            x_1 = fusion_1(x_1, feature_mem_up)
            x_2 = conv_1(x_2)
            x_out_before_fusion = torch.cat((x_1, x_2), dim=1)  # 16 通道

            # --- [修改4：移除H特征计算] ---
            # (H 特征已移至 forward 方法中)
            # intermediate_features_h = [res2x_vis, res4x_vis, res8x_vis, res16x_vis]
            # return x_out_before_fusion, intermediate_features_h
            # --- [修改4：结束] ---

            return x_out_before_fusion  # 只返回最终的可见光流特征

    def _process_ir_stream(self, x_ir, encoder, CRA1, CRA2, CRA3, CRA4, dehaze,
                           convd16x, dense_4, conv_4, fusion_4,
                           convd8x, dense_3, conv_3, fusion_3,
                           convd4x, dense_2, conv_2, fusion_2,
                           convd2x, dense_1, conv_1, fusion_1):
        """
        处理红外流（与原始 _process_stream 类似，但只返回最终特征和 CRA 特征）。
        """
        # --- 编码器和 CRA ---
        x_layer3_encoder_out, x_layer2, x_layer1, x_layer0_prepool = encoder(x_ir)
        res16x_ir = CRA1(x_layer3_encoder_out)
        res8x_ir = CRA2(x_layer2)
        res4x_ir = CRA3(x_layer1)
        res2x_ir = CRA4(x_layer0_prepool)

        # Dehaze
        in_ft = res16x_ir
        res16x_dehazed = dehaze(in_ft) + res16x_ir

        res16x_1, res16x_2 = res16x_dehazed.split([(res16x_dehazed.size(1) // 2), (res16x_dehazed.size(1) // 2)], dim=1)
        feature_mem_up = [res16x_1]

        # Decoder Stage 1
        res16x_up = convd16x(res16x_dehazed)
        res16x_up = F.interpolate(res16x_up, size=res8x_ir.size()[2:], mode='bilinear', align_corners=False)
        res8x_fused = torch.add(res16x_up, res8x_ir)
        res8x_dense = dense_4(res8x_fused) + res8x_fused
        res8x_1, res8x_2 = res8x_dense.split([(res8x_dense.size(1) // 2), (res8x_dense.size(1) // 2)], dim=1)
        res8x_1 = fusion_4(res8x_1, feature_mem_up)
        res8x_2 = conv_4(res8x_2)
        feature_mem_up.append(res8x_1)
        res8x_out = torch.cat((res8x_1, res8x_2), dim=1)

        # Decoder Stage 2
        res8x_up = convd8x(res8x_out)
        res8x_up = F.interpolate(res8x_up, size=res4x_ir.size()[2:], mode='bilinear', align_corners=False)
        res4x_fused = torch.add(res8x_up, res4x_ir)
        res4x_dense = dense_3(res4x_fused) + res4x_fused
        res4x_1, res4x_2 = res4x_dense.split([(res4x_dense.size(1) // 2), (res4x_dense.size(1) // 2)], dim=1)
        res4x_1 = fusion_3(res4x_1, feature_mem_up)
        res4x_2 = conv_3(res4x_2)
        feature_mem_up.append(res4x_1)
        res4x_out = torch.cat((res4x_1, res4x_2), dim=1)

        # Decoder Stage 3
        res4x_up = convd4x(res4x_out)
        res4x_up = F.interpolate(res4x_up, size=res2x_ir.size()[2:], mode='bilinear', align_corners=False)
        res2x_fused = torch.add(res4x_up, res2x_ir)
        res2x_dense = dense_2(res2x_fused) + res2x_fused
        res2x_1, res2x_2 = res2x_dense.split([(res2x_dense.size(1) // 2), (res2x_dense.size(1) // 2)], dim=1)
        res2x_1 = fusion_2(res2x_1, feature_mem_up)
        res2x_2 = conv_2(res2x_2)
        feature_mem_up.append(res2x_1)
        res2x_out = torch.cat((res2x_1, res2x_2), dim=1)

        # Decoder Stage 4
        res2x_up = convd2x(res2x_out)
        target_size = x_ir.size()[2:] # 使用原始红外输入尺寸
        res2x_up = F.interpolate(res2x_up, size=target_size, mode='bilinear', align_corners=False)
        x_fused = res2x_up
        x_dense = dense_1(x_fused) + x_fused
        x_1, x_2 = x_dense.split([(x_dense.size(1) // 2), (x_dense.size(1) // 2)], dim=1)
        x_1 = fusion_1(x_1, feature_mem_up)
        x_2 = conv_1(x_2)
        x_out_before_fusion = torch.cat((x_1, x_2), dim=1)

        # 返回最终解码特征和 CRA 特征 (用于计算不一致性)
        cra_features = [res16x_ir, res8x_ir, res4x_ir, res2x_ir]
        return x_out_before_fusion, cra_features

    def forward(self, x_vis, x_ir):
        """
        [修改后的] 前向传播:
        1. [修改3] 仅运行IR流编码器，获取CRA特征 (ir_cra_features)。
        2. 运行VIS流编码器，获取CRA特征 (resX_vis_orig)。
        3. 执行早期注意力融合：AttentionFusion(resX_vis_orig, resX_ir) -> 得到新的 resX_vis。
        4. 使用新的 resX_vis 和 resX_ir 计算不一致性权重 inf_weights_list。
        5. 将新的 resX_vis 和 inf_weights_list 传入可见光解码器。
        6. [修改3] 将可见光解码器输出 (vis_features) 直接通过 conv_output 生成图像。
        """
        # --- [修改3：仅处理红外编码器] ---
        # (不再调用 _process_ir_stream)
        ir_layer3_encoder_out, ir_layer2, ir_layer1, ir_layer0_prepool = self.encoder_ir(x_ir)
        res16x_ir = self.CRA1_ir(ir_layer3_encoder_out)
        res8x_ir = self.CRA2_ir(ir_layer2)
        res4x_ir = self.CRA3_ir(ir_layer1)
        res2x_ir = self.CRA4_ir(ir_layer0_prepool)
        # --- [修改3：结束] ---

        # --- 2. 计算原始可见光CRA特征 ---
        vis_layer3_encoder_out, vis_layer2, vis_layer1, vis_layer0_prepool = self.encoder_vis(x_vis)
        res16x_vis_orig = self.CRA1_vis(vis_layer3_encoder_out)
        res8x_vis_orig = self.CRA2_vis(vis_layer2)
        res4x_vis_orig = self.CRA3_vis(vis_layer1)
        res2x_vis_orig = self.CRA4_vis(vis_layer0_prepool)

        # --- [修改3：执行早期注意力融合] ---
        # (确保IR特征尺寸与VIS特征尺寸匹配)
        if res16x_vis_orig.shape[2:] != res16x_ir.shape[2:]:
            res16x_ir_aligned = F.interpolate(res16x_ir, size=res16x_vis_orig.shape[2:], mode='bilinear',
                                              align_corners=False)
        else:
            res16x_ir_aligned = res16x_ir
        res16x_vis = self.enc_fusion_16x(res16x_vis_orig, res16x_ir_aligned)  # 新的融合特征 (256)

        if res8x_vis_orig.shape[2:] != res8x_ir.shape[2:]:
            res8x_ir_aligned = F.interpolate(res8x_ir, size=res8x_vis_orig.shape[2:], mode='bilinear',
                                             align_corners=False)
        else:
            res8x_ir_aligned = res8x_ir
        res8x_vis = self.enc_fusion_8x(res8x_vis_orig, res8x_ir_aligned)  # (128)

        if res4x_vis_orig.shape[2:] != res4x_ir.shape[2:]:
            res4x_ir_aligned = F.interpolate(res4x_ir, size=res4x_vis_orig.shape[2:], mode='bilinear',
                                             align_corners=False)
        else:
            res4x_ir_aligned = res4x_ir
        res4x_vis = self.enc_fusion_4x(res4x_vis_orig, res4x_ir_aligned)  # (64)

        if res2x_vis_orig.shape[2:] != res2x_ir.shape[2:]:
            res2x_ir_aligned = F.interpolate(res2x_ir, size=res2x_vis_orig.shape[2:], mode='bilinear',
                                             align_corners=False)
        else:
            res2x_ir_aligned = res2x_ir
        res2x_vis = self.enc_fusion_2x(res2x_vis_orig, res2x_ir_aligned)  # (32)

        # --- 4. 使用新的融合特征计算不一致性 ---
        inf_weight_16x = f(res16x_vis, res16x_ir_aligned) * res16x_ir_aligned
        inf_weight_8x = f(res8x_vis, res8x_ir_aligned) * res8x_ir_aligned
        inf_weight_4x = f(res4x_vis, res4x_ir_aligned) * res4x_ir_aligned
        inf_weight_2x = f(res2x_vis, res2x_ir_aligned) * res2x_ir_aligned

        inf_weights_list = [inf_weight_16x, inf_weight_8x, inf_weight_4x, inf_weight_2x]

        # 5. H 特征使用融合前的原始可见光特征 (用于蒸馏)
        vis_intermediate_h_orig = [res2x_vis_orig, res4x_vis_orig, res8x_vis_orig, res16x_vis_orig]

        # 6. 将新的早期融合特征传入解码器
        fused_vis_cra_list = [res16x_vis, res8x_vis, res4x_vis, res2x_vis]

        # --- 7. [修改3] 调用可见光解码器 ---
        vis_features = self._process_vis_decoder(
            fused_vis_cra_list, inf_weights_list,
            self.dehaze_vis,
            self.convd16x_vis, self.dense_4_vis, self.conv_4_vis, self.fusion_4_vis,
            self.convd8x_vis, self.dense_3_vis, self.conv_3_vis, self.fusion_3_vis,
            self.convd4x_vis, self.dense_2_vis, self.conv_2_vis, self.fusion_2_vis,
            self.convd2x_vis, self.dense_1_vis, self.conv_1_vis, self.fusion_1_vis,
            x_vis.size()[2:]  # 传入原始目标尺寸
        )
        # --- [修改3：结束] ---

        # --- [修改3：修改最终融合] ---
        # (删除 self.final_fusion)
        # fused_attended_features = self.final_fusion(vis_features, ir_features)

        # (直接使用 vis_features (16通道) 作为输出)
        output = self.conv_output(vis_features)  # conv_output 现在是 16 -> 3
        # --- [修改3：结束] ---

        # --- 8. 使用原始H特征计算蒸馏输出 ---
        vis_h_features = [self.H4_vis(vis_intermediate_h_orig[0]),
                          self.H3_vis(vis_intermediate_h_orig[1]),
                          self.H2_vis(vis_intermediate_h_orig[2]),
                          self.H1_vis(vis_intermediate_h_orig[3])]
        # --- [修改3：结束] ---

        return output, vis_h_features

# --- 主函数测试部分 (保持不变) ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 实例化新模型
    net = VIFNetInconsistencyTeacher().to(device)
    dummy_input_vis = torch.randn(1, 3, 256, 256).to(device)
    dummy_input_ir = torch.randn(1, 3, 256, 256).to(device)
    output_tensor, intermediate_features = net(dummy_input_vis, dummy_input_ir)

    print("Output shape:", output_tensor.shape)
    print("Intermediate features shapes:")
    for feat in intermediate_features:
        print(feat.shape)

    try:
        pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print("Total_params: ==> {}".format(pytorch_total_params))
    except Exception as e:
        print(f"Error calculating total parameters: {e}")