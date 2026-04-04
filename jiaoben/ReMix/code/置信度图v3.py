import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import transforms

# ==========================================
# 1. 骨干网络 (Res2Net-101)
# ==========================================
# 负责从图像中提取高维特征
try:
    from timm.models.res2net import res2net101_26w_4s

    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print("请确保已安装 timm 库: pip install timm")


class Res2NetBackbone(nn.Module):
    def __init__(self, model_path=None):
        super(Res2NetBackbone, self).__init__()
        # 初始化 Res2Net-101
        self.model = res2net101_26w_4s(pretrained=False)

        # 加载预训练权重并处理可能的命名不匹配
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            # 兼容性重命名处理
            new_state_dict = {k.replace('downsample.1', 'downsample.0').replace('downsample.2', 'downsample.1'): v
                              for k, v in state_dict.items()}
            self.model.load_state_dict(new_state_dict, strict=False)

    def forward(self, x):
        # 提取 Layer 3 的特征层
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        x = self.model.maxpool(x)
        l1 = self.model.layer1(x)
        l2 = self.model.layer2(l1)
        l3 = self.model.layer3(l2)
        return l3


# ==========================================
# 2. 梯度置信度计算模块 (核心逻辑)
# ==========================================
class GradientConfidenceModule(nn.Module):
    def __init__(self):
        super(GradientConfidenceModule, self).__init__()
        # 定义 Sobel 算子用于提取梯度
        kernel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        kernel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

        # 使用 register_buffer，权重会随模型移动到 GPU，且不会被视作训练参数
        self.register_buffer('k_x', torch.FloatTensor(kernel_x).view(1, 1, 3, 3))
        self.register_buffer('k_y', torch.FloatTensor(kernel_y).view(1, 1, 3, 3))

    def _get_gradient(self, x):
        """步骤 1: Gradient Filter (梯度滤波)"""
        # 对多通道特征图求平均，转化为单通道空间特征 (B, 1, H, W)
        x_mean = torch.mean(x, dim=1, keepdim=True)
        # 卷积计算梯度
        grad_x = F.conv2d(x_mean, self.k_x, padding=1)
        grad_y = F.conv2d(x_mean, self.k_y, padding=1)
        # 梯度幅值: sqrt(gx^2 + gy^2)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

    def forward(self, f_vis, f_ir):
        # 1. 分别对可见光和红外特征进行梯度滤波
        g_vis = self._get_gradient(f_vis)
        g_ir = self._get_gradient(f_ir)

        # 2. Difference Compute (差异计算)
        diff = g_vis - g_ir

        # 3. abs (绝对值) + exp (指数增强)
        # 这里的数学公式为: output = exp(|G_vis - G_ir|)
        # 它可以极大地增强两个模态之间的显著性差异
        conf_map = torch.exp(torch.abs(diff))

        return g_vis, g_ir, conf_map


# ==========================================
# 3. 执行逻辑与保存结果
# ==========================================
def cv_imread_chinese(path, mode=cv2.IMREAD_COLOR):
    """支持中文路径的图片读取"""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), mode)


def save_heatmap(tensor, output_path, cmap='jet'):
    """将特征图转换为伪彩色热力图并保存"""
    # 插值回 448x448 方便观察
    img = F.interpolate(tensor, size=(448, 448), mode='bilinear').squeeze().cpu().numpy()
    # 归一化到 [0, 1] 用于绘图
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    plt.imsave(output_path, img, cmap=cmap)


def run_pipeline(vis_path, ir_path, weight_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 预处理转换 (ImageNet 标准)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 1. 读取数据
    img_vis = cv_imread_chinese(vis_path)
    img_ir = cv_imread_chinese(ir_path, cv2.IMREAD_GRAYSCALE)  # 通常红外为单通道

    # 准备 Tensor 输入 (B, C, H, W)
    vis_t = transform(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)).unsqueeze(0)
    ir_t = transform(cv2.cvtColor(img_ir, cv2.COLOR_GRAY2RGB)).unsqueeze(0)

    # 2. 模型初始化
    backbone = Res2NetBackbone(weight_path).eval()
    module = GradientConfidenceModule().eval()

    # 3. 推理计算
    with torch.no_grad():
        f_vis = backbone(vis_t)
        f_ir = backbone(ir_t)
        g_vis, g_ir, conf_map = module(f_vis, f_ir)

    # 4. 按照流程图节点保存结果
    save_heatmap(g_vis, os.path.join(output_dir, "Step1_Vis_Gradient.jpg"), cmap='viridis')
    save_heatmap(g_ir, os.path.join(output_dir, "Step1_IR_Gradient.jpg"), cmap='viridis')
    save_heatmap(conf_map, os.path.join(output_dir, "Step3_Final_Confidence.jpg"), cmap='jet')

    print(f"✅ 处理完成！结果已保存至: {output_dir}")


# ==========================================
# 4. 路径配置
# ==========================================
if __name__ == "__main__":
    # --- 请根据你的本地环境修改以下路径 ---
    WEIGHT_PATH = r'F:\CVPR_2026_code_data\quwu_daima_xiugai_shanchuan_github\model\imagenet_model\res2net101_v1b_26w_4s-0812c246.pth'
    VIS_INPUT = r'F:\ACMMM2026\论文里的图优化\ReMix方案图优化\RGB_IR\hazy_RGB\image1.png'
    IR_INPUT = r'F:\ACMMM2026\论文里的图优化\ReMix方案图优化\RGB_IR\IR\image1.png'
    OUTPUT_DIR = r'F:\ACMMM2026\论文里的图优化\ReMix方案图优化\RGB_IR\流程图输出结果2'

    run_pipeline(VIS_INPUT, IR_INPUT, WEIGHT_PATH, OUTPUT_DIR)