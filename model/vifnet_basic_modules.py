import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# (来自 B: models/basic.py)
def Conv_B(in_channels, out_channels, kernel_size, stride=1, bias=False):
    """ (来自 B: models/basic.py) """
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), stride=stride, bias=bias)

class PALayer(nn.Module):
    """ (来自 B: models/basic.py) """
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 16, 1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 16, 1, 1, padding=0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    """ (来自 B: models/basic.py) """
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 16, 1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 16, channel, 1, padding=0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class CPAB(nn.Module):
    """ (来自 B: models/basic.py) """
    def __init__(self, dim, kernel_size, bias):
        super(CPAB, self).__init__()
        self.conv1 = Conv_B(dim, dim, kernel_size, bias=bias)
        self.act1 = nn.PReLU()
        self.conv2 = Conv_B(dim, dim, kernel_size, bias=bias)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)
    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res

class DownSample_B(nn.Module):
    """ (来自 B: models/basic.py) """
    def __init__(self, in_channels, out_channel):
        super(DownSample_B, self).__init__()
        self.conv = Conv_B(in_channels, out_channel, 1, stride=1, bias=False)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x = self.conv(x)
        return x

class UpSample_B(nn.Module):
    """ (来自 B: models/basic.py) """
    def __init__(self, in_channels, out_channel):
        super(UpSample_B, self).__init__()
        self.conv = Conv_B(in_channels, out_channel, 1, stride=1, bias=False)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv(x)
        return x

class Encoder_B(nn.Module):
    """ (来自 B: models/basic.py) """
    def __init__(self, n_feat, kernel_size, bias, atten):
        super(Encoder_B, self).__init__()
        self.atten = atten
        self.encoder_level1 =  CPAB(n_feat, kernel_size, bias=bias)
        self.encoder_level2 =  CPAB(n_feat*2, kernel_size, bias=bias)
        self.encoder_level3 =  CPAB(n_feat*4, kernel_size, bias=bias)
        self.down12  = DownSample_B(n_feat, n_feat*2)
        self.down23  = DownSample_B(n_feat*2, n_feat*4)
        if self.atten:  # feature attention
            self.atten_conv1 = Conv_B(n_feat, n_feat, 1, bias=bias)
            self.atten_conv2 = Conv_B(n_feat*2, n_feat*2, 1, bias=bias)
            self.atten_conv3 = Conv_B(n_feat*4, n_feat*4, 1, bias=bias)

    def forward(self, x, encoder_outs=None):
        if encoder_outs is None:
            enc1 = self.encoder_level1(x)
            x = self.down12(enc1)
            enc2 = self.encoder_level2(x)
            x = self.down23(enc2)
            enc3 = self.encoder_level3(x)
            return [enc1, enc2, enc3]
        else:
            # 这是代码库 B 用于注入权重的逻辑
            # [注意] 你的新方案是在代码库 A 的 Res2Net 中注入，
            # 所以这个 else 分支在你的新架构中不会被使用。
            enc1 = self.encoder_level1(x)
            enc1_fuse_nir = enc1 + self.atten_conv1(encoder_outs[0])
            x = self.down12(enc1_fuse_nir)
            enc2 = self.encoder_level2(x)
            enc2_fuse_nir = enc2 + self.atten_conv2(encoder_outs[1])
            x = self.down23(enc2_fuse_nir)
            enc3 = self.encoder_level3(x)
            enc3_fuse_nir = enc3 + self.atten_conv3(encoder_outs[2])
            return [enc1_fuse_nir, enc2_fuse_nir, enc3_fuse_nir]

class Decoder_B(nn.Module):
    """ (来自 B: models/basic.py) """
    def __init__(self, n_feat, kernel_size, bias, residual=True):
        super(Decoder_B, self).__init__()
        self.residual = residual
        self.decoder_level1 = CPAB(n_feat, kernel_size, bias=bias)
        self.decoder_level2 = CPAB(n_feat*2, kernel_size, bias=bias)
        self.decoder_level3 = CPAB(n_feat*4, kernel_size, bias=bias)
        self.skip_conv_1 = Conv_B(n_feat, n_feat, kernel_size, bias=bias)
        self.skip_conv_2 = Conv_B(n_feat*2, n_feat*2, kernel_size, bias=bias)
        self.up21  = UpSample_B(n_feat*2, n_feat)
        self.up32  = UpSample_B(n_feat*4, n_feat*2)

    def forward(self, outs):
        enc1, enc2, enc3 = outs
        dec3 = self.decoder_level3(enc3)
        x = self.up32(dec3)
        if self.residual:
            x += self.skip_conv_2(enc2)
        dec2 = self.decoder_level2(x)
        x = self.up21(dec2)
        if self.residual:
            x += self.skip_conv_1(enc1)
        dec1 = self.decoder_level1(x)
        return [dec1, dec2, dec3] # 返回 [64, 128, 256] 通道的解码器特征