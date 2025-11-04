import torch
import torch.nn as nn
from model.vifnet_basic_modules import CPAB, DownSample_B, Conv_B

# (来自 B: models/DSFE.py)
class fusion1(nn.Module):
    """ (来自 B: models/DSFE.py) """
    def __init__(self, n_feat, scale_unetfeats, kernel_size, bias):
        super(fusion1, self).__init__()
        self.conv_1 = CPAB(n_feat, kernel_size, bias=bias)
        self.conv_2 = CPAB(n_feat, kernel_size, bias=bias)
    def forward(self, x, y): # y 在这个实现中未使用
        res = self.conv_2(self.conv_1(x))
        return res

class fusion2(nn.Module):
    """ (来自 B: models/DSFE.py) """
    def __init__(self, n_feat, scale_unetfeats, bias):
        super(fusion2, self).__init__()
        self.conv_1 = Conv_B(n_feat, n_feat, 1, stride=1, bias=bias)
        self.conv_2 = Conv_B(n_feat, n_feat, 1, stride=1, bias=bias)
        # scale_unetfeats 是下一层特征图的通道数
        # H/8 (128) -> H/4 (64),  scale_unetfeats = 64, n_feat = 128. n_feat-scale = 64
        # H/16 (256) -> H/8 (128), scale_unetfeats = 128, n_feat = 256. n_feat-scale = 128
        self.downsample = DownSample_B(n_feat-scale_unetfeats, n_feat)
    def forward(self, x, y):
        x = self.conv_1(x)
        x += self.downsample(y)
        res = self.conv_2(x)
        return res

class DSFE(nn.Module):
    """ (来自 B: models/DSFE.py) """
    def __init__(self, n_feat, kernel_size, bias):
        super(DSFE, self).__init__()

        relu = nn.PReLU()

        # 对应 64, 128, 256 通道
        self.enc_1 = nn.Sequential(nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias), relu)
        self.enc_2 = nn.Sequential(nn.Conv2d(n_feat*2, n_feat*2, kernel_size=1, bias=bias), relu)
        self.enc_3 = nn.Sequential(nn.Conv2d(n_feat*4, n_feat*4, kernel_size=1, bias=bias), relu)

        self.dec_1 = nn.Sequential(nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias), relu)
        self.dec_2 = nn.Sequential(nn.Conv2d(n_feat*2, n_feat*2, kernel_size=1, bias=bias), relu)
        self.dec_3 = nn.Sequential(nn.Conv2d(n_feat*4, n_feat*4, kernel_size=1, bias=bias), relu)

        self.fs_1 = fusion1(n_feat, n_feat, kernel_size, bias)
        self.fs_2 = fusion2(n_feat*2, n_feat, bias)
        self.fs_3 = fusion2(n_feat*4, n_feat*2, bias)

        self.sigmoid_1 = nn.Sequential(Conv_B(n_feat, n_feat, kernel_size, bias=bias), nn.Sigmoid())
        self.sigmoid_2 = nn.Sequential(Conv_B(n_feat*2, n_feat*2, kernel_size, bias=bias), nn.Sigmoid())
        self.sigmoid_3 = nn.Sequential(Conv_B(n_feat*4, n_feat*4, kernel_size, bias=bias), nn.Sigmoid())

    def forward(self, encoder_feature, decoder_feature):
        enc1, enc2, enc3 = encoder_feature # [64, 128, 256]
        dec1, dec2, dec3 = decoder_feature # [64, 128, 256]

        feat1 = self.enc_1(enc1) + self.dec_1(dec1)
        stru1 = self.fs_1(feat1, None) # (64)
        feat2 = self.enc_2(enc2) + self.dec_2(dec2)
        stru2 = self.fs_2(feat2, stru1) # (128)
        feat3 = self.enc_3(enc3) + self.dec_3(dec3)
        stru3 = self.fs_3(feat3, stru2) # (256)

        Stru1 = self.sigmoid_1(stru1)
        Stru2 = self.sigmoid_2(stru2)
        Stru3 = self.sigmoid_3(stru3)

        return [Stru1, Stru2, Stru3] # 返回 [64, 128, 256] 通道的结构图