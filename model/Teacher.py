"""
这段PyTorch代码定义了一个名为 Teacher (教师) 的复杂深度学习模型架构，这很可能是一个用于图像去雾（Image Dehazing）的高性能网络。
该模型的核心是一个编码器-解码器（Encoder-Decoder）结构：编码器部分使用了强大的 Res2Net-101（通过 Res2Net 和 Bottle2neck 类实现）作为主干网络，
并加载了在 ImageNet 上预训练的权重，以提取输入图像的多尺度特征。解码器部分则是一个非常复杂的上采样路径，它逐层（从x16到x1）融合来自编码器的对应层级特征（跳跃连接），
并在这个过程中使用了残差密集块（RDB）和自定义的多尺度解码器融合块（MDCBlock）来精炼和融合特征。最终，该 Teacher 模型不仅会输出一个3通道的去雾后图像，
还会额外返回一组由 H 模块生成的、来自不同解码器层级的中间特征图，这表明该模型是为知识蒸馏（Knowledge Distillation）而设计的，这些中间特征将用于指导一个（未在此文件中定义的）学生模型进行学习。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange  # 导入einops库，用于张量操作（此文件中未显式使用）
import math


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
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
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
        print(f'input={x.size()}')
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        print(f'after maxpool: {x.size()}')

        x = self.layer1(x)
        print(f'after layer1: {x.size}')
        x = self.layer2(x)
        print(f'after layer2: {x.size}')
        x = self.layer3(x)
        print(f'after layer3: {x.size}')
        x = self.layer4(x)
        print(f'after layer4: {x.size}')

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        print(f'x: {x.size}')
        x = self.fc(x)

        print(f'after fc output: {x.size}')
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
        说明：
        - 与 Pre_Res2Net 不同，去除了分类头，仅保留特征提取部分（到 layer3）。
        - 返回多个尺度的特征图，供解码器使用。
        - 结构包括初始卷积层、最大池化层和三个阶段的 Bottle2neck 块。
    """

    def __init__(self, block, layers, baseWidth=26, scale=4):
        """
                初始化 Res2Net 编码器。
                参数：
                - block: Bottle2neck 类。
                - layers: 列表，指定每个阶段的块数量，例如 [3, 4, 23]。
                - baseWidth: 控制 Bottle2neck 中每组通道的基础宽度，默认 26。
                - scale: 控制 Bottle2neck 中特征图分组数量，默认 4。
        """
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        # 初始卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
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
        # Res2Net的三个阶段（去雾模型中通常只用前几个阶段作为编码器）
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
        # 初始化下采样层（downsample），用于残差连接的形状匹配
        downsample = None
        # 检查是否需要下采样层：1) stride != 1（空间下采样）；2) 输入通道数 != 输出通道数
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 下采样层结构：
            # 1. AvgPool2d: 通过平均池化实现空间下采样，保持与 Bottle2neck 的 stride 一致
            # 2. Conv2d: 1x1 卷积调整通道数，从 self.inplanes 到 planes * block.expansion
            # 3. BatchNorm2d: 归一化输出通道
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride,
                             ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        # 初始化层列表，用于存储所有 Bottle2neck 块
        layers = []
        # 添加第一个 Bottle2neck 块，可能是下采样块（如果 stride != 1 或需要通道调整）
        # 参数说明：
        # - self.inplanes: 当前输入通道数
        # - planes: 基础输出通道数
        # - stride: 控制空间下采样
        # - downsample: 下采样层，用于残差连接
        # - stype='stage': 表示这是阶段的第一个块，可能包含池化
        # - baseWidth, scale: 控制 Bottle2neck 的多尺度特性
        layers.append(block(self.inplanes, planes, stride, downsample=downsample,
                            stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        # 更新输入通道数为当前阶段的输出通道数（planes * expansion）
        self.inplanes = planes * block.expansion
        # 添加后续的 Bottle2neck 块（无下采样，stride=1）
        # 这些块直接处理 self.inplanes 通道的输入，输出通道保持不变
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x_layer0 = x  # 保存conv1后的特征
        x = self.maxpool(x)

        x_layer1 = self.layer1(x)  # layer1 输出
        x_layer2 = self.layer2(x_layer1)  # layer2 输出
        x_layer3 = self.layer3(x_layer2)  # layer3 输出

        # 返回多尺度特征图，用于解码器
        # x_layer3: torch.Size([1, 1024, 16, 16]) (假设输入256x256)
        # x_layer2: torch.Size([1, 512, 32, 32])
        # x_layer1: torch.Size([1, 256, 64, 64])
        # x_layer0: torch.Size([1, 64, 128, 128])
        return x_layer3, x_layer2, x_layer1, x_layer0


class ConvBlock(torch.nn.Module):
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


# Residual dense block (RDB) architecture
class RDB(nn.Module):
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

# --- 新增 DualStreamTeacher 模型 ---
class DualStreamTeacher(nn.Module):
    """
        DualStreamTeacher: 双流去雾模型（可见光流 + 红外流），支持知识蒸馏。
        说明：
        - 包含两个独立的编码器-解码器流：可见光流和红外流，分别处理可见光和红外图像。
        - 每个流基于 Res2Net 编码器，解码器包含上采样、残差密集块（RDB）和多尺度融合块（MDCBlock）。
        - 最终通过特征拼接和卷积层融合两流特征，生成去雾图像。
        - 可见光流额外生成中间特征（H 层输出），用于知识蒸馏，指导学生模型。
        """
    def __init__(self, res_blocks=18):
        """
                初始化 DualStreamTeacher 模型。
                参数：
                - res_blocks: 去雾模块中的残差块数量，默认 18。
                """
        super(DualStreamTeacher, self).__init__()

        # --- 可见光流 ---
        # 编码器 (共享 Res2Net 定义, 但实例化两次)
        self.encoder_vis = Res2Net(Bottle2neck, [3, 4, 23], baseWidth=26, scale=4) # 注意，原代码Res2Net只到layer3

        # 尝试加载预训练权重到可见光编码器
        try:
            # 注意 Pre_Res2Net 和 Res2Net 在原始代码中的区别，这里我们只用Res2Net到layer3
            # 如果要加载完整预训练模型，需要调整Res2Net定义或加载逻辑
            res2net101_full = nn.Module() # 创建一个临时模块来加载完整权重
            res2net101_full.load_state_dict(torch.load('D:/liu_lan_qi_xia_zai/CoA-main_daima_xiugai/model/imagenet_model/res2net101_v1b_26w_4s-0812c246.pth', map_location='cpu'), strict=False)
            pretrained_dict = res2net101_full.state_dict()

            # 过滤掉不匹配的键 (例如 layer4, fc, avgpool)
            model_dict = self.encoder_vis.state_dict()
            key_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(key_dict)
            self.encoder_vis.load_state_dict(model_dict)
            print("Successfully loaded pretrained weights for visible stream encoder.")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights for visible stream encoder. {e}")


        # CRA 和 H 层 (可见光)
        self.CRA1_vis = nn.Conv2d(1024, 256, kernel_size=1) # 对应 layer3_encoder_out
        self.CRA2_vis = nn.Conv2d(512, 128, kernel_size=1)  # 对应 layer2
        self.CRA3_vis = nn.Conv2d(256, 64, kernel_size=1)   # 对应 layer1
        self.CRA4_vis = nn.Conv2d(64, 32, kernel_size=1)    # 对应 layer0_prepool (原始x_layer3的地方)

        self.H1_vis = nn.Conv2d(256, 128, kernel_size=1)
        self.H2_vis = nn.Conv2d(128, 64, kernel_size=1)
        self.H3_vis = nn.Conv2d(64, 32, kernel_size=1)
        self.H4_vis = nn.Conv2d(32, 16, kernel_size=1)

        # Dehaze 模块 (可见光)
        self.dehaze_vis = nn.Sequential()
        for i in range(0, res_blocks):
            self.dehaze_vis.add_module('res%d' % i, ResidualBlock(256)) # 输入通道为CRA1_vis的输出通道

        # 解码器模块 (可见光)
        self.convd16x_vis = UpsampleConvLayer(256, 128, kernel_size=3, stride=2) # 输出通道改为128以匹配dense_4
        self.dense_4_vis = nn.Sequential(ResidualBlock(128), ResidualBlock(128), ResidualBlock(128))
        self.conv_4_vis = RDB(64, 4, 64) # RDB的输入通道是dense输出的一半
        self.fusion_4_vis = Decoder_MDCBlock1(64, 2, mode='iter2') # 输入通道是dense输出的一半

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


        # --- 红外流 ---
        # 编码器 (新实例，结构相同)
        self.encoder_ir = Res2Net(Bottle2neck, [3, 4, 23], baseWidth=26, scale=4) # 结构与encoder_vis相同

        # CRA 和 H 层 (红外) - 注意通道数与可见光流对应层一致
        self.CRA1_ir = nn.Conv2d(1024, 256, kernel_size=1)
        self.CRA2_ir = nn.Conv2d(512, 128, kernel_size=1)
        self.CRA3_ir = nn.Conv2d(256, 64, kernel_size=1)
        self.CRA4_ir = nn.Conv2d(64, 32, kernel_size=1)

        # 注意：蒸馏通常只针对主任务流（可见光），所以红外流可能不需要H层
        # 如果需要对红外流也做蒸馏，则需要添加 self.H1_ir 等

        # Dehaze 模块 (红外)
        self.dehaze_ir = nn.Sequential()
        for i in range(0, res_blocks):
            self.dehaze_ir.add_module('res%d' % i, ResidualBlock(256))

        # 解码器模块 (红外) - 结构与可见光流相同
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


        # --- 融合与输出 ---
        # 融合前可见光输出是16通道，红外也是16通道
        # 拼接后是32通道，所以输出卷积输入通道改为32
        self.conv_output = ConvLayer(32, 3, kernel_size=3, stride=1)


    def _process_stream(self, x, encoder, CRA1, CRA2, CRA3, CRA4, dehaze,
                        convd16x, dense_4, conv_4, fusion_4,
                        convd8x, dense_3, conv_3, fusion_3,
                        convd4x, dense_2, conv_2, fusion_2,
                        convd2x, dense_1, conv_1, fusion_1):

        """
                辅助函数，处理单一流（可见光或红外）的编码、去雾和解码。
                参数：
                - x: 输入图像张量。
                - encoder, CRA1-4, dehaze, convd*, dense_*, conv_*, fusion_*: 对应模块。
                返回：
                - x_out_before_fusion: 解码器最终输出特征。
                - intermediate_features: 中间特征列表，用于知识蒸馏。
                """

        x_layer3_encoder_out, x_layer2, x_layer1, x_layer0_prepool = encoder(x)

        # CRA 处理
        res16x = CRA1(x_layer3_encoder_out) # (B, 256, H/16, W/16)
        res8x = CRA2(x_layer2)             # (B, 128, H/8, W/8)
        res4x = CRA3(x_layer1)             # (B, 64, H/4, W/4)
        res2x = CRA4(x_layer0_prepool)     # (B, 32, H/2, W/2) - 注意这里原始代码用的是 layer3，但根据尺度应该是 layer0_prepool

        # Dehaze 处理最深层特征
        res_dehaze_in = res16x
        # 原始代码的 in_ft = res16x * 2 可能不太合理，改为直接使用 res16x
        # 或者可以尝试简单的残差连接 in_ft = res16x
        in_ft = res16x # 简化输入
        res16x_dehazed = dehaze(in_ft) + res16x # 使用残差连接

        res16x_1, res16x_2 = res16x_dehazed.split([(res16x_dehazed.size(1) // 2), (res16x_dehazed.size(1) // 2)], dim=1)
        feature_mem_up = [res16x_1] # (B, 128, H/16, W/16)

        # 解码 stage 1 (16x -> 8x)
        res16x_up = convd16x(res16x_dehazed) # (B, 128, H/8, W/8)
        # 调整尺寸以匹配 res8x
        res16x_up = F.interpolate(res16x_up, size=res8x.size()[2:], mode='bilinear', align_corners=False)
        res8x_fused = torch.add(res16x_up, res8x) # (B, 128, H/8, W/8)
        res8x_dense = dense_4(res8x_fused) + res8x_fused # 简化残差连接

        res8x_1, res8x_2 = res8x_dense.split([(res8x_dense.size(1) // 2), (res8x_dense.size(1) // 2)], dim=1) # (B, 64, H/8, W/8) each
        res8x_1 = fusion_4(res8x_1, feature_mem_up) # (B, 64, H/8, W/8)
        res8x_2 = conv_4(res8x_2) # (B, 64, H/8, W/8)
        feature_mem_up.append(res8x_1)
        res8x_out = torch.cat((res8x_1, res8x_2), dim=1) # (B, 128, H/8, W/8)

        # 解码 stage 2 (8x -> 4x)
        res8x_up = convd8x(res8x_out) # (B, 64, H/4, W/4)
        res8x_up = F.interpolate(res8x_up, size=res4x.size()[2:], mode='bilinear', align_corners=False)
        res4x_fused = torch.add(res8x_up, res4x) # (B, 64, H/4, W/4)
        res4x_dense = dense_3(res4x_fused) + res4x_fused

        res4x_1, res4x_2 = res4x_dense.split([(res4x_dense.size(1) // 2), (res4x_dense.size(1) // 2)], dim=1) # (B, 32, H/4, W/4) each
        res4x_1 = fusion_3(res4x_1, feature_mem_up) # (B, 32, H/4, W/4)
        res4x_2 = conv_3(res4x_2) # (B, 32, H/4, W/4)
        feature_mem_up.append(res4x_1)
        res4x_out = torch.cat((res4x_1, res4x_2), dim=1) # (B, 64, H/4, W/4)

        # 解码 stage 3 (4x -> 2x)
        res4x_up = convd4x(res4x_out) # (B, 32, H/2, W/2)
        res4x_up = F.interpolate(res4x_up, size=res2x.size()[2:], mode='bilinear', align_corners=False)
        res2x_fused = torch.add(res4x_up, res2x) # (B, 32, H/2, W/2)
        res2x_dense = dense_2(res2x_fused) + res2x_fused

        res2x_1, res2x_2 = res2x_dense.split([(res2x_dense.size(1) // 2), (res2x_dense.size(1) // 2)], dim=1) # (B, 16, H/2, W/2) each
        res2x_1 = fusion_2(res2x_1, feature_mem_up) # (B, 16, H/2, W/2)
        res2x_2 = conv_2(res2x_2) # (B, 16, H/2, W/2)
        feature_mem_up.append(res2x_1)
        res2x_out = torch.cat((res2x_1, res2x_2), dim=1) # (B, 32, H/2, W/2)

        # 解码 stage 4 (2x -> 1x)
        res2x_up = convd2x(res2x_out) # (B, 16, H, W)
        res2x_up = F.interpolate(res2x_up, size=x.size()[2:], mode='bilinear', align_corners=False)
        # 原始代码这里是 torch.add(res2x, res2x)，应该是一个笔误，改为与上采样结果相加
        # 另外，这里没有来自输入的原始分辨率特征可以加，所以只用上采样结果
        x_fused = res2x_up # (B, 16, H, W)
        x_dense = dense_1(x_fused) + x_fused

        x_1, x_2 = x_dense.split([(x_dense.size(1) // 2), (x_dense.size(1) // 2)], dim=1) # (B, 8, H, W) each
        x_1 = fusion_1(x_1, feature_mem_up) # (B, 8, H, W)
        x_2 = conv_1(x_2) # (B, 8, H, W)
        x_out_before_fusion = torch.cat((x_1, x_2), dim=1) # (B, 16, H, W)

        # 返回处理后的特征以及用于蒸馏的中间特征（如果需要）
        # 返回 CRA 处理后的特征用于可能的蒸馏
        intermediate_features = [res2x, res4x, res8x, res16x] # 按 H 层的顺序返回？ H4(res2x), H3(res4x), H2(res8x), H1(res16x)
        return x_out_before_fusion, intermediate_features


    def forward(self, x_vis, x_ir):
        """
                前向传播，处理可见光和红外图像，生成去雾图像和中间特征。
                参数：
                - x_vis: 可见光输入图像，形状为 (batch_size, 3, H, W)。
                - x_ir: 红外输入图像，形状为 (batch_size, 3, H, W)。
                返回：
                - output: 去雾后图像，形状为 (batch_size, 3, H, W)。
                - vis_h_features: 可见光流的中间特征列表，用于知识蒸馏。
                """
        # 处理可见光流
        vis_features, vis_intermediate = self._process_stream(
            x_vis, self.encoder_vis, self.CRA1_vis, self.CRA2_vis, self.CRA3_vis, self.CRA4_vis, self.dehaze_vis,
            self.convd16x_vis, self.dense_4_vis, self.conv_4_vis, self.fusion_4_vis,
            self.convd8x_vis, self.dense_3_vis, self.conv_3_vis, self.fusion_3_vis,
            self.convd4x_vis, self.dense_2_vis, self.conv_2_vis, self.fusion_2_vis,
            self.convd2x_vis, self.dense_1_vis, self.conv_1_vis, self.fusion_1_vis
        )

        # 处理红外流
        ir_features, _ = self._process_stream(  # 红外流的中间特征通常不用于蒸馏
            x_ir, self.encoder_ir, self.CRA1_ir, self.CRA2_ir, self.CRA3_ir, self.CRA4_ir, self.dehaze_ir,
            self.convd16x_ir, self.dense_4_ir, self.conv_4_ir, self.fusion_4_ir,
            self.convd8x_ir, self.dense_3_ir, self.conv_3_ir, self.fusion_3_ir,
            self.convd4x_ir, self.dense_2_ir, self.conv_2_ir, self.fusion_2_ir,
            self.convd2x_ir, self.dense_1_ir, self.conv_1_ir, self.fusion_1_ir
        )

        # 融合特征 (简单拼接)
        fused_features = torch.cat((vis_features, ir_features), dim=1) # (B, 16+16, H, W) = (B, 32, H, W)

        # 输出层
        output = self.conv_output(fused_features) # (B, 3, H, W)

        # 计算可见光流的 H 特征用于蒸馏
        vis_h_features = [self.H4_vis(vis_intermediate[0]), # res2xx
                          self.H3_vis(vis_intermediate[1]), # res4xx
                          self.H2_vis(vis_intermediate[2]), # res8xx
                          self.H1_vis(vis_intermediate[3])] # res16xx

        # 返回去雾后的可见光图像和可见光流的中间特征
        return output, vis_h_features


# --- 主函数测试部分 (可选, 用于验证模型是否能运行) ---
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 修改测试代码以使用新模型
    net = DualStreamTeacher().to(device)
    # 输入两个图像
    dummy_input_vis = torch.randn(1, 3, 256, 256).to(device)
    dummy_input_ir = torch.randn(1, 3, 256, 256).to(device)
    output_tensor, intermediate_features = net(dummy_input_vis, dummy_input_ir)

    print("Output shape:", output_tensor.shape)
    print("Intermediate features shapes:")
    for feat in intermediate_features:
        print(feat.shape)

    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total_params: ==> {}".format(pytorch_total_params))












































# 原始代码
# class Teacher(nn.Module):
#     """
#     教师模型 (Teacher Model)
#     这是一个基于Res2Net编码器和复杂解码器的图像去雾网络。
#     """
#
#     def __init__(self, res_blocks=18):
#         super(Teacher, self).__init__()
#
#         # --- 1. 编码器 ---
#         self.encoder = Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4)
#
#         # --- 2. 加载预训练权重 ---
#         # 实例化一个用于加载权重的Pre_Res2Net（分类模型）
#         res2net101 = Pre_Res2Net(Bottle2neck, [3, 4, 23, 3], baseWidth=26, scale=4)
#         # 加载ImageNet预训练权重
#         # 原始
#         # res2net101.load_state_dict(torch.load('./model/imagenet_model/res2net101_v1b_26w_4s-0812c246.pth'))
#         #路径修改
#         res2net101.load_state_dict(torch.load('D:/liu_lan_qi_xia_zai/CoA-main/model/imagenet_model/res2net101_v1b_26w_4s-0812c246.pth'))
#         pretrained_dict = res2net101.state_dict()  # 获取预训练权重
#         model_dict = self.encoder.state_dict()  # 获取当前编码器（Res2Net）的权重字典
#         # 筛选出预训练模型中与当前编码器层名匹配的权重
#         key_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#         model_dict.update(key_dict)  # 更新编码器的权重
#         self.encoder.load_state_dict(model_dict)  # 加载筛选后的权重
#
#         # --- 3. 解码器 ---
#
#         # 3.1 瓶颈层（最深层特征处理）
#         self.dehaze = nn.Sequential()
#         for i in range(0, res_blocks):
#             self.dehaze.add_module('res%d' % i, ResidualBlock(256))  # 堆叠18个残差块
#
#         # 3.2 解码器阶段 1 (x16 -> x8)
#         self.convd16x = UpsampleConvLayer(256, 128, kernel_size=3, stride=2)  # 上采样
#         self.dense_4 = nn.Sequential(  # 残差块
#             ResidualBlock(128),
#             ResidualBlock(128),
#             ResidualBlock(128)
#         )
#         self.conv_4 = RDB(64, 4, 64)  # 残差密集块
#         self.fusion_4 = Decoder_MDCBlock1(64, 2, mode='iter2')  # 融合块 (2个层级)
#
#         # 3.3 解码器阶段 2 (x8 -> x4)
#         self.convd8x = UpsampleConvLayer(128, 64, kernel_size=3, stride=2)
#         self.dense_3 = nn.Sequential(
#             ResidualBlock(64),
#             ResidualBlock(64),
#             ResidualBlock(64)
#         )
#         self.conv_3 = RDB(32, 4, 32)
#         self.fusion_3 = Decoder_MDCBlock1(32, 3, mode='iter2')  # 融合块 (3个层级)
#
#         # 3.4 解码器阶段 3 (x4 -> x2)
#         self.convd4x = UpsampleConvLayer(64, 32, kernel_size=3, stride=2)
#         self.dense_2 = nn.Sequential(
#             ResidualBlock(32),
#             ResidualBlock(32),
#             ResidualBlock(32)
#         )
#         self.conv_2 = RDB(16, 4, 16)
#         self.fusion_2 = Decoder_MDCBlock1(16, 4, mode='iter2')  # 融合块 (4个层级)
#
#         # 3.5 解码器阶段 4 (x2 -> x1)
#         self.convd2x = UpsampleConvLayer(32, 16, kernel_size=3, stride=2)
#         self.dense_1 = nn.Sequential(
#             ResidualBlock(16),
#             ResidualBlock(16),
#             ResidualBlock(16)
#         )
#         self.conv_1 = RDB(8, 4, 8)
#         self.fusion_1 = Decoder_MDCBlock1(8, 5, mode='iter2')  # 融合块 (5个层级)
#
#         # 3.6 输出层
#         self.conv_output = ConvLayer(16, 3, kernel_size=3, stride=1)  # 转换为3通道RGB图像
#
#         # --- 4. 辅助模块 ---
#         # CRA (Channel Reduction) 1x1 卷积，用于调整编码器输出特征的通道数
#         self.CRA1 = nn.Conv2d(1024, 256, kernel_size=1)
#         self.CRA2 = nn.Conv2d(512, 128, kernel_size=1)
#         self.CRA3 = nn.Conv2d(256, 64, kernel_size=1)
#         self.CRA4 = nn.Conv2d(64, 32, kernel_size=1)
#
#         # H 模块（1x1 卷积），用于生成知识蒸馏所需的中间特征
#         self.H1 = nn.Conv2d(256, 128, kernel_size=1)
#         self.H2 = nn.Conv2d(128, 64, kernel_size=1)
#         self.H3 = nn.Conv2d(64, 32, kernel_size=1)
#         self.H4 = nn.Conv2d(32, 16, kernel_size=1)
#
#     def forward(self, x):
#         ini = x  # 保存原始输入
#
#         # 1. 编码器提取多尺度特征
#         x_layer0, x_layer1, x_layer2, x_layer3 = self.encoder(x)
#         # (x_layer0=1024, x_layer1=512, x_layer2=256, x_layer3=64 通道)
#
#         # 2. 调整通道数，准备用于解码器（跳跃连接）
#         res16x = self.CRA1(x_layer0)  # [B, 256, H/16, W/16]
#         res8x = self.CRA2(x_layer1)  # [B, 128, H/8, W/8]
#         res4x = self.CRA3(x_layer2)  # [B, 64, H/4, W/4]
#         res2x = self.CRA4(x_layer3)  # [B, 32, H/2, W/2]
#
#         # 3. 生成用于知识蒸馏的中间特征
#         res16xx = self.H1(res16x)
#         res8xx = self.H2(res8x)
#         res4xx = self.H3(res4x)
#         res2xx = self.H4(res2x)
#
#         # --- 4. 解码器 ---
#
#         # 4.1 瓶颈层
#         res_dehaze = res16x
#         in_ft = res16x * 2  # 特征加倍？(可能是笔误，或者某种特征增强)
#         res16x = self.dehaze(in_ft) + in_ft - res_dehaze  # 残差结构
#
#         # 将特征在通道上分成两半
#         res16x_1, res16x_2 = res16x.split([(res16x.size()[1] // 2), (res16x.size()[1] // 2)], dim=1)
#         feature_mem_up = [res16x_1]  # 初始化多尺度融合的特征列表（用于MDCBlock）
#
#         # 4.2 解码器阶段 1 (x16 -> x8)
#         res16x = self.convd16x(res16x)  # 上采样
#         res16x = F.interpolate(res16x, res8x.size()[2:], mode='bilinear')  # 插值到x8尺度
#         res8x = torch.add(res16x, res8x)  # 融合跳跃连接
#         res8x = self.dense_4(res8x) + res8x - res16x  # 残差结构
#         # 分割特征
#         res8x_1, res8x_2 = res8x.split([(res8x.size()[1] // 2), (res8x.size()[1] // 2)], dim=1)
#         res8x_1 = self.fusion_4(res8x_1, feature_mem_up)  # 第一半通过MDCBlock融合
#         res8x_2 = self.conv_4(res8x_2)  # 第二半通过RDB
#         feature_mem_up.append(res8x_1)  # 将融合后的特征加入列表
#         res8x = torch.cat((res8x_1, res8x_2), dim=1)  # 重新拼接
#
#         # 4.3 解码器阶段 2 (x8 -> x4)
#         res8x = self.convd8x(res8x)  # 上采样
#         res8x = F.interpolate(res8x, res4x.size()[2:], mode='bilinear')  # 插值到x4尺度
#         res4x = torch.add(res8x, res4x)  # 融合跳跃连接
#         res4x = self.dense_3(res4x) + res4x - res8x  # 残差结构
#         # 分割特征
#         res4x_1, res4x_2 = res4x.split([(res4x.size()[1] // 2), (res4x.size()[1] // 2)], dim=1)
#         res4x_1 = self.fusion_3(res4x_1, feature_mem_up)  # 第一半通过MDCBlock融合 (此时 feature_mem_up 有2个元素)
#         res4x_2 = self.conv_3(res4x_2)  # 第二半通过RDB
#         feature_mem_up.append(res4x_1)  # 加入列表
#         res4x = torch.cat((res4x_1, res4x_2), dim=1)  # 拼接
#
#         # 4.4 解码器阶段 3 (x4 -> x2)
#         res4x = self.convd4x(res4x)  # 上采样
#         res4x = F.interpolate(res4x, res2x.size()[2:], mode='bilinear')  # 插值到x2尺度
#         res2x = torch.add(res4x, res2x)  # 融合跳跃连接
#         res2x = self.dense_2(res2x) + res2x - res4x  # 残差结构
#         # 分割特征
#         res2x_1, res2x_2 = res2x.split([(res2x.size()[1] // 2), (res2x.size()[1] // 2)], dim=1)
#         res2x_1 = self.fusion_2(res2x_1, feature_mem_up)  # 第一半通过MDCBlock融合
#         res2x_2 = self.conv_2(res2x_2)  # 第二半通过RDB
#         feature_mem_up.append(res2x_1)  # 加入列表
#         res2x = torch.cat((res2x_1, res2x_2), dim=1)  # 拼接
#
#         # 4.5 解码器阶段 4 (x2 -> x1)
#         res2x = self.convd2x(res2x)  # 上采样
#         res2x = F.interpolate(res2x, ini.size()[2:], mode='bilinear')  # 插值到x1（原始）尺度
#         x = torch.add(res2x, res2x)  # (?? 为什么是 res2x + res2x ? 可能是笔误，应为融合原始输入ini或其浅层特征)
#         x = self.dense_1(x) + x - res2x  # 残差结构
#         # 分割特征
#         x_1, x_2 = x.split([(x.size()[1] // 2), (x.size()[1] // 2)], dim=1)
#         x_1 = self.fusion_1(x_1, feature_mem_up)  # 第一半通过MDCBlock融合
#         x_2 = self.conv_1(x_2)  # 第二半通过RDB
#         x = torch.cat((x_1, x_2), dim=1)  # 拼接
#
#         # 4.6 输出
#         x = self.conv_output(x)  # 最终输出3通道图像
#
#         # 返回去雾图像 和 用于蒸馏的中间特征列表
#         return x, [res2xx, res4xx, res8xx, res16xx]
#
#
# if __name__ == "__main__":
#     # 测试代码
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     net = Teacher().to(device)  # 实例化教师模型
#     dummy_input = torch.randn(1, 3, 256, 256).to(device)  # 创建一个虚拟输入
#     output_tensor = net(dummy_input)  # 前向传播
#     print(output_tensor[0].shape)  # 打印输出图像的尺寸
#     # 计算并打印模型的可训练参数量
#     pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
#     print("Total_params: ==> {}".format(pytorch_total_params))