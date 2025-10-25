import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange  # 导入einops库，用于张量维度重排（此文件中未使用）
import math


class ConvBlock(torch.nn.Module):
    """
    标准卷积块 (Conv -> Norm -> Activation)
    """

    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        # 归一化层
        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        # 激活层
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
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation != 'no':  # 'no' 表示不使用激活函数
            return self.act(out)
        else:
            return out


class DeconvBlock(torch.nn.Module):
    """
    标准反卷积（转置卷积）块 (Deconv -> Norm -> Activation)
    """

    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu',
                 norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        # 归一化层
        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        # 激活层
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
    多尺度解码器/融合块 (MDCBlock)。
    用于融合来自不同层级（尺度）的特征。
    """

    def __init__(self, num_filter, num_ft, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None,
                 mode='iter1'):
        super(Decoder_MDCBlock1, self).__init__()
        self.mode = mode  # 融合模式 (如 'iter1', 'iter2' 等)
        self.num_ft = num_ft - 1  # 特征层级的数量
        self.down_convs = nn.ModuleList()  # 下采样卷积列表
        self.up_convs = nn.ModuleList()  # 上采样反卷积列表
        # 根据层级数，创建对应的下采样（Conv）和上采样（Deconv）卷积
        # 通道数随着层级加深而加倍
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
        :param ft_h: 当前层级（高层，High-level）的特征
        :param ft_l_list: 之前所有（低层，Low-level）特征的列表
        """
        if self.mode == 'iter1' or self.mode == 'conv':
            # 模式1
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
            ft_fusion = ft_h  # 融合结果初始化为当前特征
            for i in range(len(ft_l_list)):  # 遍历所有低层特征
                ft = ft_fusion
                # 1. 将当前融合特征下采样到与 ft_l_list[i] 相同的尺度
                for j in range(self.num_ft - i):
                    ft = self.down_convs[j](ft)

                ft = ft - ft_l_list[i]  # 2. 计算差异

                # 3. 将差异上采样回原始尺度
                for j in range(self.num_ft - i):
                    ft = self.up_convs[self.num_ft - i - j - 1](ft)

                ft_fusion = ft_fusion + ft  # 4. 将差异（校正）加回到融合特征上

        if self.mode == 'iter3':
            # 模式3
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
            # 模式4
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


class Encoder_MDCBlock1(torch.nn.Module):
    """
    多尺度编码器/融合块 (MDCBlock)。
    与 Decoder_MDCBlock1 结构对称，用于编码器阶段的特征融合。
    """

    def __init__(self, num_filter, num_ft, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None,
                 mode='iter1'):
        super(Encoder_MDCBlock1, self).__init__()
        self.mode = mode
        self.num_ft = num_ft - 1
        self.up_convs = nn.ModuleList()  # 上采样反卷积列表
        self.down_convs = nn.ModuleList()  # 下采样卷积列表
        # 注意：这里的通道数变化与Decoder相反，是逐级减半
        for i in range(self.num_ft):
            self.up_convs.append(
                DeconvBlock(num_filter // (2 ** i), num_filter // (2 ** (i + 1)), kernel_size, stride, padding, bias,
                            activation, norm=None)
            )
            self.down_convs.append(
                ConvBlock(num_filter // (2 ** (i + 1)), num_filter // (2 ** i), kernel_size, stride, padding, bias,
                          activation, norm=None)
            )

    def forward(self, ft_l, ft_h_list):
        """
        :param ft_l: 当前层级（低层，Low-level）的特征
        :param ft_h_list: 之前所有（高层，High-level）特征的列表
        """
        if self.mode == 'iter1' or self.mode == 'conv':
            # 模式1
            ft_l_list = []
            for i in range(len(ft_h_list)):
                ft_l_list.append(ft_l)
                ft_l = self.up_convs[self.num_ft - len(ft_h_list) + i](ft_l)

            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft_fusion = self.down_convs[self.num_ft - i - 1](ft_fusion - ft_h_list[i]) + ft_l_list[
                    len(ft_h_list) - i - 1]

        if self.mode == 'iter2':
            # 模式2：(代码中使用的模式)
            ft_fusion = ft_l  # 融合结果初始化为当前特征
            for i in range(len(ft_h_list)):  # 遍历所有高层特征
                ft = ft_fusion
                # 1. 将当前融合特征上采样到与 ft_h_list[i] 相同的尺度
                for j in range(self.num_ft - i):
                    ft = self.up_convs[j](ft)

                ft = ft - ft_h_list[i]  # 2. 计算差异

                # 3. 将差异下采样回原始尺度
                for j in range(self.num_ft - i):
                    # print(j)
                    ft = self.down_convs[self.num_ft - i - j - 1](ft)

                ft_fusion = ft_fusion + ft  # 4. 将差异（校正）加回到融合特征上

        if self.mode == 'iter3':
            # 模式3
            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft = ft_fusion
                for j in range(i + 1):
                    ft = self.up_convs[j](ft)
                ft = ft - ft_h_list[len(ft_h_list) - i - 1]
                for j in range(i + 1):
                    # print(j)
                    ft = self.down_convs[i + 1 - j - 1](ft)
                ft_fusion = ft_fusion + ft

        if self.mode == 'iter4':
            # 模式4
            ft_fusion = ft_l
            for i in range(len(ft_h_list)):
                ft = ft_l
                for j in range(self.num_ft - i):
                    ft = self.up_convs[j](ft)
                ft = ft - ft_h_list[i]
                for j in range(self.num_ft - i):
                    # print(j)
                    ft = self.down_convs[self.num_ft - i - j - 1](ft)
                ft_fusion = ft_fusion + ft

        return ft_fusion


class make_dense(nn.Module):
    """
    密集连接块的单层实现（用于RDB）
    """

    def __init__(self, nChannels, growthRate, kernel_size=3):
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
    残差密集块 (Residual Dense Block)
    """

    def __init__(self, nChannels, nDenselayer, growthRate, scale=1.0):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        self.scale = scale
        modules = []
        for i in range(nDenselayer):  # 堆叠多个 'make_dense' 层
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate  # 通道数随层数增加
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)  # 1x1卷积，恢复通道数

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out) * self.scale  # 1x1卷积，并乘以一个缩放因子
        out = out + x  # 添加残差连接 (Short Skip Connection)
        return out


class ConvLayer(nn.Module):
    """
    带反射填充 (ReflectionPad) 的卷积层
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(torch.nn.Module):
    """
    上采样层，使用 'nearest-exact' 插值 + 1x1 卷积
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(UpsampleConvLayer, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        # 计算插值后的目标高宽
        h = (x.shape[2] - 1) * self.stride + self.kernel_size
        w = (x.shape[3] - 1) * self.stride + self.kernel_size
        x = F.interpolate(x, size=(h, w), mode="nearest-exact")  # 最近邻插值
        out = self.conv2d(x)  # 1x1 卷积
        return out


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    """
    默认卷积层，自动计算padding以保持分辨率
    """
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class ResidualBlock(torch.nn.Module):
    """
    标准残差块 (Conv -> PReLU -> Conv)
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = default_conv(channels, channels, 3)
        self.conv2 = default_conv(channels, channels, 3)
        self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1  # 残差缩放
        out = torch.add(out, residual)  # 添加残差
        return out


class Student_x(nn.Module):
    """
    学生模型 (Student Model)
    这是一个轻量级的编码器-解码器结构，用于知识蒸馏。
    它使用了标准残差块 (ResidualBlock)，而不是上一版本中的DEConv。
    """

    def __init__(self, res_blocks=1):
        super(Student_x, self).__init__()

        # --- 编码器 (Encoder) ---

        # 1. 初始卷积
        self.conv_input = ConvLayer(3, 8, kernel_size=11, stride=1)  # 3 -> 8
        self.dense0 = nn.Sequential(
            ResidualBlock(8)
        )

        # 2. 阶段 1 (x1 -> x2)
        self.conv2x = ConvLayer(8, 16, kernel_size=3, stride=2)  # 下采样 8 -> 16
        self.conv1 = RDB(8, 4, 8)  # RDB 处理一半特征 (通道8)
        self.fusion1 = Encoder_MDCBlock1(8, 2, mode='iter2')  # MDCBlock 融合另一半 (通道8)
        self.dense1 = nn.Sequential(
            ResidualBlock(16)
        )

        # 3. 阶段 2 (x2 -> x4)
        self.conv4x = ConvLayer(16, 32, kernel_size=3, stride=2)  # 下采样 16 -> 32
        self.conv2 = RDB(16, 4, 16)  # RDB (通道16)
        self.fusion2 = Encoder_MDCBlock1(16, 3, mode='iter2')  # MDCBlock (通道16)
        self.dense2 = nn.Sequential(
            ResidualBlock(32)
        )

        # 4. 阶段 3 (x4 -> x8)
        self.conv8x = ConvLayer(32, 64, kernel_size=3, stride=2)  # 下采样 32 -> 64
        self.conv3 = RDB(32, 4, 32)
        self.fusion3 = Encoder_MDCBlock1(32, 4, mode='iter2')
        self.dense3 = nn.Sequential(
            ResidualBlock(64)
        )

        # 5. 阶段 4 (x8 -> x16)
        self.conv16x = ConvLayer(64, 128, kernel_size=3, stride=2)  # 下采样 64 -> 128
        self.conv4 = RDB(64, 4, 64)
        self.fusion4 = Encoder_MDCBlock1(64, 5, mode='iter2')

        # --- 解码器 (Decoder) ---

        # 6. 瓶颈层 (Bottleneck)
        self.dehaze = nn.Sequential()
        for i in range(0, res_blocks):
            self.dehaze.add_module('res%d' % i, ResidualBlock(128))

        # 7. 解码器阶段 1 (x16 -> x8)
        self.convd16x = UpsampleConvLayer(128, 64, kernel_size=3, stride=2)  # 上采样 128 -> 64
        self.dense_4 = nn.Sequential(
            ResidualBlock(64)
        )
        self.conv_4 = RDB(32, 4, 32)
        self.fusion_4 = Decoder_MDCBlock1(32, 2, mode='iter2')

        # 8. 解码器阶段 2 (x8 -> x4)
        self.convd8x = UpsampleConvLayer(64, 32, kernel_size=3, stride=2)  # 上采样 64 -> 32
        self.dense_3 = nn.Sequential(
            ResidualBlock(32)
        )
        self.conv_3 = RDB(16, 4, 16)
        self.fusion_3 = Decoder_MDCBlock1(16, 3, mode='iter2')

        # 9. 解码器阶段 3 (x4 -> x2)
        self.convd4x = UpsampleConvLayer(32, 16, kernel_size=3, stride=2)  # 上采样 32 -> 16
        self.dense_2 = nn.Sequential(
            ResidualBlock(16)
        )
        self.conv_2 = RDB(8, 4, 8)
        self.fusion_2 = Decoder_MDCBlock1(8, 4, mode='iter2')

        # 10. 解码器阶段 4 (x2 -> x1)
        self.convd2x = UpsampleConvLayer(16, 8, kernel_size=3, stride=2)  # 上采样 16 -> 8
        self.dense_1 = nn.Sequential(
            ResidualBlock(8)
        )
        self.conv_1 = RDB(4, 4, 4)
        self.fusion_1 = Decoder_MDCBlock1(4, 5, mode='iter2')

        # 11. 输出卷积
        self.conv_output = ConvLayer(8, 3, kernel_size=3, stride=1)  # 8 -> 3

    def forward(self, x):
        ini = x  # 保存原始输入 (未在此模型中直接使用)

        # --- 编码器 ---
        res1x = self.conv_input(x)
        res1x_1, res1x_2 = res1x.split([(res1x.size()[1] // 2), (res1x.size()[1] // 2)], dim=1)  # 特征分割 (4, 4)
        feature_mem = [res1x_1]  # 初始化MDCBlock的特征列表
        x = self.dense0(res1x) + res1x  # 残差连接 (输出 8 通道)

        res2x = self.conv2x(x)  # 下采样 (输出 16 通道)
        res2x_1, res2x_2 = res2x.split([(res2x.size()[1] // 2), (res2x.size()[1] // 2)], dim=1)  # 分割 (8, 8)
        res2x_1 = self.fusion1(res2x_1, feature_mem)  # 一半MDC融合 (输入8, 列表[4]) -> 输出8
        res2x_2 = self.conv1(res2x_2)  # 另一半RDB处理 (输入8) -> 输出8
        feature_mem.append(res2x_1)  # 将融合特征存入列表 (列表[4, 8])
        res2x = torch.cat((res2x_1, res2x_2), dim=1)  # 拼接 (8+8=16)
        res2x = self.dense1(res2x) + res2x

        res4x = self.conv4x(res2x)  # 下采样 (输出 32 通道)
        res4x_1, res4x_2 = res4x.split([(res4x.size()[1] // 2), (res4x.size()[1] // 2)], dim=1)  # 分割 (16, 16)
        res4x_1 = self.fusion2(res4x_1, feature_mem)  # MDC融合 (输入16, 列表[4, 8]) -> 输出16
        res4x_2 = self.conv2(res4x_2)  # RDB处理 (输入16) -> 输出16
        feature_mem.append(res4x_1)  # (列表[4, 8, 16])
        res4x = torch.cat((res4x_1, res4x_2), dim=1)
        res4x = self.dense2(res4x) + res4x

        res8x = self.conv8x(res4x)  # 下采样 (输出 64 通道)
        res8x_1, res8x_2 = res8x.split([(res8x.size()[1] // 2), (res8x.size()[1] // 2)], dim=1)  # 分割 (32, 32)
        res8x_1 = self.fusion3(res8x_1, feature_mem)  # MDC融合 (输入32, 列表[4, 8, 16]) -> 输出32
        res8x_2 = self.conv3(res8x_2)  # RDB处理 (输入32) -> 输出32
        feature_mem.append(res8x_1)  # (列表[4, 8, 16, 32])
        res8x = torch.cat((res8x_1, res8x_2), dim=1)
        res8x = self.dense3(res8x) + res8x

        res16x = self.conv16x(res8x)  # 下采样 (输出 128 通道)
        res16x_1, res16x_2 = res16x.split([(res16x.size()[1] // 2), (res16x.size()[1] // 2)], dim=1)  # 分割 (64, 64)
        res16x_1 = self.fusion4(res16x_1, feature_mem)  # MDC融合 (输入64, 列表[4, 8, 16, 32]) -> 64
        res16x_2 = self.conv4(res16x_2)  # RDB处理 (输入64) -> 64
        res16x = torch.cat((res16x_1, res16x_2), dim=1)

        # 保存编码器输出的特征，用于知识蒸馏
        res2xx = res2x
        res4xx = res4x
        res8xx = res8x
        res16xx = res16x

        # --- 解码器 ---
        res_dehaze = res16x
        in_ft = res16x * 2  # 特征加倍
        res16x = self.dehaze(in_ft) + in_ft - res_dehaze  # 瓶颈层
        res16x_1, res16x_2 = res16x.split([(res16x.size()[1] // 2), (res16x.size()[1] // 2)], dim=1)  # 分割 (64, 64)
        feature_mem_up = [res16x_1]  # 初始化解码器MDCBlock的特征列表

        # 解码器阶段 1
        res16x = self.convd16x(res16x)  # 上采样 (128 -> 64)
        res16x = F.interpolate(res16x, res8x.size()[2:], mode='bilinear')  # 插值到x8尺度
        res8x = torch.add(res16x, res8x)  # 跳跃连接 (融合编码器的 res8x)
        res8x = self.dense_4(res8x) + res8x - res16x  # 残差
        res8x_1, res8x_2 = res8x.split([(res8x.size()[1] // 2), (res8x.size()[1] // 2)], dim=1)  # 分割 (32, 32)
        res8x_1 = self.fusion_4(res8x_1, feature_mem_up)  # 一半MDC融合 (输入32, 列表[64]) -> 32
        res8x_2 = self.conv_4(res8x_2)  # 另一半RDB (输入32) -> 32
        feature_mem_up.append(res8x_1)  # (列表[64, 32])
        res8x = torch.cat((res8x_1, res8x_2), dim=1)

        # 解码器阶段 2
        res8x = self.convd8x(res8x)  # 上采样 (64 -> 32)
        res8x = F.interpolate(res8x, res4x.size()[2:], mode='bilinear')  # 插值到x4尺度
        res4x = torch.add(res8x, res4x)  # 跳跃连接 (融合编码器的 res4x)
        res4x = self.dense_3(res4x) + res4x - res8x
        res4x_1, res4x_2 = res4x.split([(res4x.size()[1] // 2), (res4x.size()[1] // 2)], dim=1)  # 分割 (16, 16)
        res4x_1 = self.fusion_3(res4x_1, feature_mem_up)  # MDC融合 (输入16, 列表[64, 32]) -> 16
        res4x_2 = self.conv_3(res4x_2)  # RDB (输入16) -> 16
        feature_mem_up.append(res4x_1)  # (列表[64, 32, 16])
        res4x = torch.cat((res4x_1, res4x_2), dim=1)

        # 解码器阶段 3
        res4x = self.convd4x(res4x)  # 上采样 (32 -> 16)
        res4x = F.interpolate(res4x, res2x.size()[2:], mode='bilinear')  # 插值到x2尺度
        res2x = torch.add(res4x, res2x)  # 跳跃连接 (融合编码器的 res2x)
        res2x = self.dense_2(res2x) + res2x - res4x
        res2x_1, res2x_2 = res2x.split([(res2x.size()[1] // 2), (res2x.size()[1] // 2)], dim=1)  # 分割 (8, 8)
        res2x_1 = self.fusion_2(res2x_1, feature_mem_up)  # MDC融合 (输入8, 列表[64, 32, 16]) -> 8
        res2x_2 = self.conv_2(res2x_2)  # RDB (输入8) -> 8
        feature_mem_up.append(res2x_1)  # (列表[64, 32, 16, 8])
        res2x = torch.cat((res2x_1, res2x_2), dim=1)

        # 解码器阶段 4
        res2x = self.convd2x(res2x)  # 上采样 (16 -> 8)
        res2x = F.interpolate(res2x, x.size()[2:], mode='bilinear')  # 插值到 x1 尺度
        x = torch.add(res2x, x)  # 跳跃连接 (与编码器第一阶段的输出`x`融合)
        x = self.dense_1(x) + x - res2x  # 残差
        x_1, x_2 = x.split([(x.size()[1] // 2), (x.size()[1] // 2)], dim=1)  # 分割 (4, 4)
        x_1 = self.fusion_1(x_1, feature_mem_up)  # MDC融合 (输入4, 列表[64, 32, 16, 8]) -> 4
        x_2 = self.conv_1(x_2)  # RDB (输入4) -> 4
        x = torch.cat((x_1, x_2), dim=1)

        # 输出
        x = self.conv_output(x)

        # 返回去雾图像 和 用于蒸馏的中间特征列表
        return x, [res2xx, res4xx, res8xx, res16xx]


if __name__ == "__main__":
    # --- 测试代码 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = Student_x().to(device)  # 实例化学生模型
    dummy_input = torch.randn(1, 3, 256, 256).to(device)  # 创建虚拟输入
    output_tensor = net(dummy_input)  # 前向传播
    print(output_tensor[0].shape)  # 打印输出图像的尺寸
    # 计算并打印模型的可训练参数量
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total_params: ==> {}".format(pytorch_total_params))