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
try:
    from timm.models.res2net import res2net101_26w_4s

    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False


class Res2NetBackbone(nn.Module):
    def __init__(self, model_path=None):
        super(Res2NetBackbone, self).__init__()
        self.model = res2net101_26w_4s(pretrained=False)
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            new_state_dict = {k.replace('downsample.1', 'downsample.0').replace('downsample.2', 'downsample.1'): v
                              for k, v in state_dict.items() if 'downsample.0' not in k or 'layer' not in k}
            self.model.load_state_dict(new_state_dict, strict=False)

    def forward(self, x):
        x = self.model.conv1(x);
        x = self.model.bn1(x);
        x = self.model.act1(x);
        x = self.model.maxpool(x)
        l1 = self.model.layer1(x);
        l2 = self.model.layer2(l1);
        l3 = self.model.layer3(l2)
        return l3


# ==========================================
# 2. 梯度置信度计算模块 (基于流程图)
# ==========================================
class GradientConfidenceModule(nn.Module):
    def __init__(self):
        super(GradientConfidenceModule, self).__init__()
        # 定义 Sobel 算子用于提取梯度
        kernel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        kernel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        self.kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0)
        self.kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)

    def get_gradient(self, x):
        # 对特征图进行通道平均处理，转化为单通道空间特征
        x_mean = torch.mean(x, dim=1, keepdim=True)
        grad_x = F.conv2d(x_mean, self.kernel_x.to(x.device), padding=1)
        grad_y = F.conv2d(x_mean, self.kernel_y.to(x.device), padding=1)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

    def forward(self, f_vis, f_ir):
        # 1. Gradient Filter
        grad_vis = self.get_gradient(f_vis)
        grad_ir = self.get_gradient(f_ir)

        # 2. Difference Compute
        diff = grad_vis - grad_ir

        # 3. abs. + exp
        # 这里的 exp 通常用于增强显著性差异，公式：output = exp(|diff|)
        conf_map = torch.exp(torch.abs(diff))

        # 为了可视化，将其归一化到 [0, 1]
        conf_map = (conf_map - conf_map.min()) / (conf_map.max() - conf_map.min() + 1e-8)

        return grad_vis, grad_ir, conf_map


# ==========================================
# 3. 执行与保存逻辑
# ==========================================
def cv_imread_chinese(path, mode=cv2.IMREAD_COLOR):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), mode)


def run_pipeline(vis_path, ir_path, weight_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # 图像预处理
    transform = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((448, 448)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_vis = cv_imread_chinese(vis_path);
    img_ir = cv_imread_chinese(ir_path, cv2.IMREAD_GRAYSCALE)
    vis_t = transform(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)).unsqueeze(0)
    ir_t = transform(cv2.cvtColor(img_ir, cv2.COLOR_GRAY2RGB)).unsqueeze(0)

    # 提取特征与计算
    backbone = Res2NetBackbone(weight_path).eval()
    module = GradientConfidenceModule().eval()

    with torch.no_grad():
        f_vis = backbone(vis_t)
        f_ir = backbone(ir_t)
        g_vis, g_ir, result_map = module(f_vis, f_ir)

    # 后处理用于保存
    def save_map(tensor, name, cmap='jet'):
        img = F.interpolate(tensor, size=(448, 448), mode='bilinear').squeeze().cpu().numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        plt.imsave(os.path.join(output_dir, f"{name}.jpg"), img, cmap=cmap)

    # 按照流程图节点输出：
    save_map(g_vis, "1_Vis_Gradient", cmap='viridis')  # 梯度滤波后的可见光
    save_map(g_ir, "2_IR_Gradient", cmap='viridis')  # 梯度滤波后的红外
    save_map(result_map, "3_Final_Confidence_Map", cmap='jet')  # abs + exp 后的最终图

    print(f"流程图节点图像已保存至: {output_dir}")


# ==========================================
# 4. 路径设置
# ==========================================
if __name__ == "__main__":
    WEIGHT = r'F:\CVPR_2026_code_data\quwu_daima_xiugai_shanchuan_github\model\imagenet_model\res2net101_v1b_26w_4s-0812c246.pth'
    VIS_IN = r'F:\ACMMM2026\论文里的图优化\ReMix方案图优化\RGB_IR\hazy_RGB\image1.png'
    IR_IN = r'F:\ACMMM2026\论文里的图优化\ReMix方案图优化\RGB_IR\IR\image1.png'
    OUT_D = r'F:\ACMMM2026\论文里的图优化\ReMix方案图优化\RGB_IR\流程图输出结果'

    run_pipeline(VIS_IN, IR_IN, WEIGHT, OUT_D)