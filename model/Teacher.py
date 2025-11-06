import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import torchvision.transforms.functional as TF
from collections import OrderedDict

# --- [修改] 使用绝对导入 ---
from model.vifnet_basic_modules import Encoder_B, Decoder_B, Conv_B, CPAB
from model.dsfe import DSFE


# --- VIFNet 不一致性函数 f(x, y) (已废弃) ---
def f(x, y):
    """VIFNet inconsistency function (code version)"""
    return (1 - x) * (1 - y) + 1 / 2 * x * y


# --- [新增] 可学习的自适应门控模块 ---
class GatedAdaptiveInjection(nn.Module):
    """
    可学习的门控自适应注入模块。
    它根据可见光置信度图 (x_vis_conf) 和红外结构 (x_ir_struct)
    来学习一个门控 (0到1)，决定应注入多少红外特征。
    """

    def __init__(self, in_channels):
        super(GatedAdaptiveInjection, self).__init__()

        # 这个小型CNN学习(VIS置信度, IR结构) -> (最佳注入权重)的复杂映射
        self.gate_generator = nn.Sequential(
            # 输入是拼接后的 (vis_conf, ir_struct)，通道数为 in_channels * 2
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 输出通道为 in_channels
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()  # 输出门控权重 (0 ~ 1)
        )

    def forward(self, x_vis_conf, x_ir_struct):
        """
        x_vis_conf: 可见光置信度图 (来自 Pass 1 DSFE_vis)
        x_ir_struct: 红外结构特征 (来自 Pass 1 DSFE_ir)
        """

        # 1. 将两个信息源拼接
        combined_info = torch.cat((x_vis_conf, x_ir_struct), dim=1)

        # 2. 学习门控权重
        learned_gate = self.gate_generator(combined_info)

        # 3. 将门控应用于红外特征
        return x_ir_struct * learned_gate


# --- [新增结束] ---


# --- 基础模块 (SobelEdgeDetector, Pre_Res2Net, Bottle2neck, Res2Net(3通道输入), ConvBlock, DeconvBlock, Decoder_MDCBlock1, make_dense, RDB, ConvLayer, UpsampleConvLayer, ResidualBlock) ---
# ... (这些基础模块的代码与上一个版本相同，这里省略以保持简洁) ...
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
        *** [修改]：支持“串联注入”不一致性权重 ***
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

        # --- [新增] 注入权重适配器 (用于 Pass 2 注入) ---
        # 匹配 DSFE(B) 输出 -> Res2Net(A) *输入* (layer 1/2/3 的输出通道)
        # DSFE [64, 128, 256] -> Res2Net Layer [256, 512, 1024]
        self.inject_conv1 = nn.Conv2d(64, 256, kernel_size=1, bias=False)  # H/4
        self.inject_conv2 = nn.Conv2d(128, 512, kernel_size=1, bias=False)  # H/8
        self.inject_conv3 = nn.Conv2d(256, 1024, kernel_size=1, bias=False)  # H/16
        # --- [新增结束] ---

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

    def forward(self, x, inf_weights=None):  # [修改] 增加 inf_weights=None 参数
        """
        [修改后] 的 Res2Net forward，支持串联注入 (Sequential Injection)

        inf_weights: 一个列表 [Stru3(256), Stru2(128), Stru1(64)]
                     对应 [H/16, H/8, H/4] 尺度
        """

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x_layer0 = x  # (B, 64, H/2, W/2) - 用于 H4 蒸馏 和 H/2 解码
        x_maxpool = self.maxpool(x)  # (B, 64, H/4, W/4)

        # --- H/4 尺度 ---
        x_layer1_orig = self.layer1(x_maxpool)  # (B, 256, H/4, W/4)
        if inf_weights is not None:
            inf_w_4 = self.inject_conv1(inf_weights[2])  # 64 -> 256
            inf_w_4 = F.interpolate(inf_w_4, size=x_layer1_orig.shape[2:], mode='bilinear', align_corners=False)
            x_layer1_fused = x_layer1_orig + inf_w_4
        else:
            x_layer1_fused = x_layer1_orig

        # --- H/8 尺度 ---
        x_layer2_orig = self.layer2(x_layer1_fused)  # (B, 512, H/8, W/8)
        if inf_weights is not None:
            inf_w_8 = self.inject_conv2(inf_weights[1])  # 128 -> 512
            inf_w_8 = F.interpolate(inf_w_8, size=x_layer2_orig.shape[2:], mode='bilinear', align_corners=False)
            x_layer2_fused = x_layer2_orig + inf_w_8
        else:
            x_layer2_fused = x_layer2_orig

        # --- H/16 尺度 ---
        x_layer3_orig = self.layer3(x_layer2_fused)  # (B, 1024, H/16, W/16)
        if inf_weights is not None:
            inf_w_16 = self.inject_conv3(inf_weights[0])  # 256 -> 1024
            inf_w_16 = F.interpolate(inf_w_16, size=x_layer3_orig.shape[2:], mode='bilinear', align_corners=False)
            x_layer3_fused = x_layer3_orig + inf_w_16
        else:
            x_layer3_fused = x_layer3_orig

        # [修改] 返回 注入后(fused)的特征（用于解码）和 注入前(orig)的特征（用于蒸馏）
        fused_outputs = [x_layer3_fused, x_layer2_fused, x_layer1_fused, x_layer0]
        original_outputs = [x_layer3_orig, x_layer2_orig, x_layer1_orig, x_layer0]

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
        """
                前向传播，执行 RELU 激活和通道拼接。
                参数：
                - x: 输入张量。
                返回：
                - out: 拼接后的张量 (输入通道 + growthRate 通道)。
        """
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


# --- [删除] ChannelAttentionFusion 模块 ---
# (上一版中用于最终融合，现已删除)


# --- 修改后的 VIFNetInconsistencyTeacher 模型 ---
class VIFNetInconsistencyTeacher(nn.Module):
    """
    [修改后] 双流教师模型，采用“两阶段精炼”架构。
    阶段一：使用 VIFnet 轻量级模块提取结构特征。
    阶段二：使用 Res2Net 重量级模块进行去雾精炼。
    """

    def __init__(self, res_blocks=18):
        super(VIFNetInconsistencyTeacher, self).__init__()

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

        # --- [新增] 自适应门控注入模块 (GAI) ---
        self.gai_h4 = GatedAdaptiveInjection(in_channels=b_n_feat)  # H/4 尺度 (64 C)
        self.gai_h8 = GatedAdaptiveInjection(in_channels=b_n_feat * 2)  # H/8 尺度 (128 C)
        self.gai_h16 = GatedAdaptiveInjection(in_channels=b_n_feat * 4)  # H/16 尺度 (256 C)
        # --- [新增结束] ---

        # --- 阶段二 (Pass 2) 模块 (来自代码库 A) ---

        # --- 可见光流 (主网络) ---
        # [修改]：Res2Net 现在内部包含了 inject_conv 模块
        self.encoder_vis = Res2Net(Bottle2neck, [3, 4, 23], baseWidth=26, scale=4, in_channels=3)
        # ... (加载权重代码保留) ...
        try:
            res2net101_full = Pre_Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4)
            # [请注意]：请确保你本地 'D:/...' 路径下存在此文件
            pretrained_path = 'D:/liu_lan_qi_xia_zai/CoA-main_daima_xiugai_jiehe_ir_edge_xiugai_teacher_v5/model/imagenet_model/res2net101_v1b_26w_4s-0812c246.pth'
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

        # --- [删除] 红外流 (Pass 2) 模块 ---
        # self.encoder_ir = ... (已删除)
        # ... (所有 _ir 后缀的解码器模块均已删除) ...

        # --- [删除] 最终融合模块 ---
        # self.final_fusion = ... (已删除)

        # --- [修改] 最终输出卷积 ---
        # 原: self.conv_output = ConvLayer(32, 3, kernel_size=3, stride=1)
        # 新:
        self.conv_output = ConvLayer(16, 3, kernel_size=3, stride=1)

    # --- [删除] _process_vis_decoder 和 _process_ir_stream ---
    # (这两个函数的功能已被内联并重构到新的 forward 方法中)

    # --- [重写] forward 方法 ---
    def forward(self, x_vis, x_ir, haze_mask=None):
        """
        [修改后] 的前向传播

        参数:
            x_vis (Tensor): 有雾可见光图像
            x_ir (Tensor): 红外图像
            haze_mask (Tensor, 可选):
                一个 [B, 1, H, W] 的真值掩码 (0=清晰, 1=有雾)。
                如果提供，将用作指导；如果为 None (默认)，将使用可学习的 GAI 模块。
        """

        # --- 阶段一 & 二：并行结构提取 (Pass 1 - B 模块) ---

        # 1a. VIS 流 (Pass 1) -> DSFE_vis
        vis_b_fea1 = self.vis_layer1_b(x_vis)
        vis_b_enc_features = self.encoder_b_vis(vis_b_fea1)  # [64, 128, 256]
        vis_b_dec_features = self.decoder_b_vis(vis_b_enc_features)  # [64, 128, 256]
        # vis_structure 列表顺序: [H/4(64), H/8(128), H/16(256)]
        vis_structure = self.dsfe_vis(vis_b_enc_features, vis_b_dec_features)

        # 1b. IR 流 (Pass 1) -> DSFE_ir
        ir_b_fea1 = self.ir_layer1_b(x_ir)
        ir_b_enc_features = self.encoder_b_ir(ir_b_fea1)  # [64, 128, 256]
        ir_b_dec_features = self.decoder_b_ir(ir_b_enc_features)  # [64, 128, 256]
        # ir_structure 列表顺序: [H/4(64), H/8(128), H/16(256)]
        ir_structure = self.dsfe_ir(ir_b_enc_features, ir_b_dec_features)

        # --- 阶段三：计算自适应注入权重 ---

        # inf_weight_list 注入列表顺序：[H/16(256), H/8(128), H/4(64)]
        inf_weight_list = [None, None, None]

        if haze_mask is None:
            # --- 模式 A: 训练 或 无掩码测试 (使用可学习的门控) ---

            # H/16 尺度 (vis_structure[2], ir_structure[2])
            inf_weight_list[0] = self.gai_h16(vis_structure[2], ir_structure[2])

            # H/8 尺度 (vis_structure[1], ir_structure[1])
            inf_weight_list[1] = self.gai_h8(vis_structure[1], ir_structure[1])

            # H/4 尺度 (vis_structure[0], ir_structure[0])
            inf_weight_list[2] = self.gai_h4(vis_structure[0], ir_structure[0])

        else:
            # --- 模式 B: 有掩码的测试 (使用真值掩码) ---
            # (掩码 = 1 表示有雾，应注入 IR；掩码 = 0 表示清晰，不注入)

            # H/16 尺度
            mask_h16 = F.interpolate(haze_mask, size=ir_structure[2].shape[2:], mode='bilinear', align_corners=False)
            inf_weight_list[0] = ir_structure[2] * mask_h16

            # H/8 尺度
            mask_h8 = F.interpolate(haze_mask, size=ir_structure[1].shape[2:], mode='bilinear', align_corners=False)
            inf_weight_list[1] = ir_structure[1] * mask_h8

            # H/4 尺度
            mask_h4 = F.interpolate(haze_mask, size=ir_structure[0].shape[2:], mode='bilinear', align_corners=False)
            inf_weight_list[2] = ir_structure[0] * mask_h4

        # --- 阶段四：精炼编码与注入（Pass 2 - A 模块）---

        # 4a. 运行 Pass 2 Encoder (代码库 A) 并进行串联注入
        fused_outputs, original_outputs = self.encoder_vis(x_vis, inf_weight_list)

        x_layer3_fused, x_layer2_fused, x_layer1_fused, x_layer0 = fused_outputs
        x_layer3_orig, x_layer2_orig, x_layer1_orig, _ = original_outputs

        # [用于蒸馏的 H 特征]
        vis_h_features = [
            self.H4_vis(self.CRA4_vis(x_layer0)),  # 32 -> 16
            self.H3_vis(self.CRA3_vis(x_layer1_orig)),  # 64 -> 32
            self.H2_vis(self.CRA2_vis(x_layer2_orig)),  # 128 -> 64
            self.H1_vis(self.CRA1_vis(x_layer3_orig))  # 256 -> 128
        ]

        # --- 阶段五：最终解码（Pass 2 - A 模块）---

        # 5a. CRA 降维 (输入是已融合的特征)
        res16x_vis = self.CRA1_vis(x_layer3_fused)
        res8x_vis = self.CRA2_vis(x_layer2_fused)
        res4x_vis = self.CRA3_vis(x_layer1_fused)
        res2x_vis = self.CRA4_vis(x_layer0)

        # 5b. 运行代码库 A 的 VIS 解码器
        in_ft = res16x_vis
        res16x_dehazed = self.dehaze_vis(in_ft) + res16x_vis
        res16x_1, res16x_2 = res16x_dehazed.split([(res16x_dehazed.size(1) // 2), (res16x_dehazed.size(1) // 2)], dim=1)
        feature_mem_up = [res16x_1]

        # Stage 1 (H/16 -> H/8)
        res16x_up = self.convd16x_vis(res16x_dehazed)
        res16x_up = F.interpolate(res16x_up, size=res8x_vis.size()[2:], mode='bilinear', align_corners=False)
        res8x_fused = torch.add(res16x_up, res8x_vis)
        res8x_dense = self.dense_4_vis(res8x_fused) + res8x_fused
        res8x_1, res8x_2 = res8x_dense.split([(res8x_dense.size(1) // 2), (res8x_dense.size(1) // 2)], dim=1)
        res8x_1 = self.fusion_4_vis(res8x_1, feature_mem_up)
        res8x_2 = self.conv_4_vis(res8x_2)
        feature_mem_up.append(res8x_1)
        res8x_out = torch.cat((res8x_1, res8x_2), dim=1)

        # Stage 2 (H/8 -> H/4)
        res8x_up = self.convd8x_vis(res8x_out)
        res8x_up = F.interpolate(res8x_up, size=res4x_vis.size()[2:], mode='bilinear', align_corners=False)
        res4x_fused = torch.add(res8x_up, res4x_vis)
        res4x_dense = self.dense_3_vis(res4x_fused) + res4x_fused
        res4x_1, res4x_2 = res4x_dense.split([(res4x_dense.size(1) // 2), (res4x_dense.size(1) // 2)], dim=1)
        res4x_1 = self.fusion_3_vis(res4x_1, feature_mem_up)
        res4x_2 = self.conv_3_vis(res4x_2)
        feature_mem_up.append(res4x_1)
        res4x_out = torch.cat((res4x_1, res4x_2), dim=1)

        # Stage 3 (H/4 -> H/2)
        res4x_up = self.convd4x_vis(res4x_out)
        res4x_up = F.interpolate(res4x_up, size=res2x_vis.size()[2:], mode='bilinear', align_corners=False)
        res2x_fused = torch.add(res4x_up, res2x_vis)
        res2x_dense = self.dense_2_vis(res2x_fused) + res2x_fused
        res2x_1, res2x_2 = res2x_dense.split([(res2x_dense.size(1) // 2), (res2x_dense.size(1) // 2)], dim=1)
        res2x_1 = self.fusion_2_vis(res2x_1, feature_mem_up)
        res2x_2 = self.conv_2_vis(res2x_2)
        feature_mem_up.append(res2x_1)
        res2x_out = torch.cat((res2x_1, res2x_2), dim=1)

        # Stage 4 (H/2 -> H)
        res2x_up = self.convd2x_vis(res2x_out)
        res2x_up = F.interpolate(res2x_up, size=x_vis.size()[2:], mode='bilinear', align_corners=False)
        x_fused = res2x_up
        x_dense = self.dense_1_vis(x_fused) + x_fused
        x_1, x_2 = x_dense.split([(x_dense.size(1) // 2), (x_dense.size(1) // 2)], dim=1)
        x_1 = self.fusion_1_vis(x_1, feature_mem_up)
        x_2 = self.conv_1_vis(x_2)
        vis_features = torch.cat((x_1, x_2), dim=1)  # (16 通道)

        # 5c. [运行 IR 流 (Pass 2)] --- (已删除) ---

        # 5d. 最终融合 (A 库逻辑)
        # fused_attended_features = self.final_fusion(vis_features, ir_features) # (已删除)
        output = self.conv_output(vis_features)  # (修改: 16 -> 3)

        return output, vis_h_features


# --- 主函数测试部分 (保持不变) ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 实例化新模型
    net = VIFNetInconsistencyTeacher().to(device)
    dummy_input_vis = torch.randn(1, 3, 256, 256).to(device)
    dummy_input_ir = torch.randn(1, 3, 256, 256).to(device)

    # 测试模式 A: 不带掩码 (训练模式)
    output_tensor_train, _ = net(dummy_input_vis, dummy_input_ir, haze_mask=None)
    print("Test Mode A (Training/No Mask) Output shape:", output_tensor_train.shape)

    # 测试模式 B: 带掩码 (引导测试)
    dummy_mask = torch.rand(1, 1, 256, 256).to(device)  # 模拟一个掩码
    output_tensor_test, _ = net(dummy_input_vis, dummy_input_ir, haze_mask=dummy_mask)
    print("Test Mode B (Guided Mask) Output shape:", output_tensor_test.shape)

    try:
        pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print("Total_params: ==> {}".format(pytorch_total_params))
    except Exception as e:
        print(f"Error calculating total parameters: {e}")