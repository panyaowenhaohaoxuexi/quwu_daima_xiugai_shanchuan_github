# loss/perceptual.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.cr import Vgg19  # 从 cr.py 导入 Vgg19


class PerceptualLoss(nn.Module):
    """
    VGG感知损失，包含内容损失(Content Loss)和风格损失(Style Loss)。
    """

    def __init__(self, content_weight=1.0, style_weight=1.0):
        super(PerceptualLoss, self).__init__()
        # [修改] 自动将 VGG 加载到正确的设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vgg = Vgg19().to(self.device)
        self.criterion = nn.L1Loss()
        self.content_weight = content_weight
        self.style_weight = style_weight

        # VGG 归一化参数 (与 ContrastLoss 一致)
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1).to(self.device)

        # 定义使用 VGG 的哪些层
        # 我们使用 relu1_1, relu2_1, relu3_1, relu4_1, relu5_1
        # 对应 Vgg19 的 slice1 到 slice5
        self.feature_weights = [1.0, 1.0, 1.0, 1.0, 1.0]

    def gram_matrix(self, features):
        """ 计算 Gram 矩阵 (用于风格损失) """
        n, c, h, w = features.size()
        features = features.view(n, c, h * w)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram

    def forward(self, pred_img, target_img):
        """
        pred_img: 预测图像 (模型输出)
        target_img: 目标图像 (清晰的可见光真值)
        """

        # 归一化输入
        pred_norm = (pred_img - self.mean) / self.std
        target_norm = (target_img - self.mean) / self.std

        # 提取 VGG 特征
        pred_features = self.vgg(pred_norm)

        # target_features 不计算梯度
        with torch.no_grad():
            target_features = self.vgg(target_norm)

        loss_content = 0.0
        loss_style = 0.0

        for i in range(len(pred_features)):
            # 1. 计算内容损失 (Perceptual Loss)
            # 比较特征图本身
            if self.content_weight > 0:
                loss_content += self.feature_weights[i] * self.criterion(
                    pred_features[i],
                    target_features[i].detach()
                )

            # 2. 计算风格损失 (Style Loss)
            # 比较特征图的 Gram 矩阵
            if self.style_weight > 0:
                loss_style += self.feature_weights[i] * self.criterion(
                    self.gram_matrix(pred_features[i]),
                    self.gram_matrix(target_features[i]).detach()
                )

        total_loss = (self.content_weight * loss_content) + (self.style_weight * loss_style)
        return total_loss