import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, p=2, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        """
        predict: (B, C, H, W) or (B, 1, H, W) - 预测的边缘图
        target: (B, C, H, W) or (B, 1, H, W) - 真实的边缘图 (Ground Truth)
        """
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"

        # 确保输入是 [0, 1] 范围
        # Canny 输出的是梯度幅值，可能大于1，这里将其压缩到 0-1 更有利于 Dice 计算
        # 如果你确信输入已经在 0-1 之间，可以注释掉下面这行
        predict = torch.sigmoid(predict)

        # 展平张量 (B, -1)
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        # 计算交集和并集
        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        # Dice Loss = 1 - Dice Coefficient
        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))