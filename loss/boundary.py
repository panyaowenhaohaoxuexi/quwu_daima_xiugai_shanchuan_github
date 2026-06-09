import torch
import torch.nn as nn
import torch.nn.functional as F


class BoundarySmoothnessLoss(nn.Module):
    """
    边界平滑损失：仅在 HAPM 掩码边界环形带上惩罚去雾输出图梯度，
    使补全区(M=1)与融合区(M=0)过渡自然；GT 边缘处放松约束以保留真实边缘。
    """
    def __init__(self, band_k=5, lambda_edge=10.0, eps=1e-6):
        super(BoundarySmoothnessLoss, self).__init__()
        self.band_k = band_k
        self.lambda_edge = lambda_edge
        self.eps = eps

    def _grad_xy(self, img):
        # 各向同性梯度，通道平均 → (B,1,H,W)
        gx = torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1])
        gy = torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :])
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = F.pad(gy, (0, 0, 0, 1))
        return (gx + gy).mean(dim=1, keepdim=True)

    def forward(self, pred_image, m_hard, gt_image=None):
        """
        pred_image: (B,3,H,W) 去雾输出
        m_hard:     (B,1,H,W) HAPM 硬掩码 ∈{0,1}；None 时损失为 0
        gt_image:   (B,3,H,W) GT；提供时启用边缘感知加权，None 时退化纯 TV
        """
        if m_hard is None:
            return torch.tensor(0.0, device=pred_image.device)

        # 边界环形带 = dilate(M) - erode(M)，detach（仅作空间选择器）
        m = m_hard.detach()
        pad = self.band_k // 2
        dilate = F.max_pool2d(m, kernel_size=self.band_k, stride=1, padding=pad)
        erode = -F.max_pool2d(-m, kernel_size=self.band_k, stride=1, padding=pad)
        band = dilate - erode  # (B,1,H,W) ∈{0,1}

        g_pred = self._grad_xy(pred_image)

        if gt_image is not None:
            with torch.no_grad():
                g_gt = self._grad_xy(gt_image)
                w_edge = torch.exp(-self.lambda_edge * g_gt)
        else:
            w_edge = torch.ones_like(g_pred)

        numer = (band * w_edge * g_pred).sum()
        denom = band.sum() + self.eps
        return numer / denom
