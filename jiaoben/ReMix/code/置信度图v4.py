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
except ImportError:
    print("请安装 timm: pip install timm")


class Res2NetBackbone(nn.Module):
    def __init__(self, model_path=None):
        super(Res2NetBackbone, self).__init__()
        self.model = res2net101_26w_4s(pretrained=False)
        if model_path and os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            new_state_dict = {k.replace('downsample.1', 'downsample.0').replace('downsample.2', 'downsample.1'): v
                              for k, v in state_dict.items()}
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
# 2. 空间决策场模块 (Spatial Decision Module)
# ==========================================
class SpatialDecisionModule(nn.Module):
    def __init__(self, in_channels=1024):
        super(SpatialDecisionModule, self).__init__()
        # 梯度路径算子 (Sobel)
        kernel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        kernel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        self.register_buffer('k_x', torch.FloatTensor(kernel_x).view(1, 1, 3, 3))
        self.register_buffer('k_y', torch.FloatTensor(kernel_y).view(1, 1, 3, 3))

        # 学习路径 (Learned Visible Cue)
        self.cue_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def _get_gradient(self, x):
        x_mean = torch.mean(x, dim=1, keepdim=True)
        grad_x = F.conv2d(x_mean, self.k_x, padding=1)
        grad_y = F.conv2d(x_mean, self.k_y, padding=1)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

    def forward(self, f_vis, f_ir):
        # A. 梯度置信度: exp(|G_vis - G_ir|)
        g_vis = self._get_gradient(f_vis)
        g_ir = self._get_gradient(f_ir)
        conf_grad = torch.exp(torch.abs(g_vis - g_ir))

        # B. 学习路径 (Visible Cue)
        vis_cue = self.sigmoid(self.cue_conv(f_vis))

        # C. 最终决策场 (SDF)
        sdf = conf_grad * vis_cue
        return g_vis, g_ir, conf_grad, vis_cue, sdf


# ==========================================
# 3. 图像保存逻辑 (包含带 Colorbar 的总图)
# ==========================================
def normalize_tensor(tensor):
    """将 Tensor 转换为 448x448 的 [0,1] 归一化 numpy 数组"""
    img = F.interpolate(tensor, size=(448, 448), mode='bilinear').squeeze().cpu().numpy()
    img_min, img_max = img.min(), img.max()
    return (img - img_min) / (img_max - img_min + 1e-8)


def save_map_as_jet(tensor, path):
    """保存纯净的 Jet 颜色分布图"""
    img_norm = normalize_tensor(tensor)
    plt.imsave(path, img_norm, cmap='jet')


def save_master_map_with_colorbar(tensor, path, title="Spatial Decision Field"):
    """保存带有颜色强度柱状图（Colorbar）的总图"""
    img_norm = normalize_tensor(tensor)

    plt.figure(figsize=(10, 8))
    # 显示图像
    im = plt.imshow(img_norm, cmap='jet')
    # 添加标题 (可选)
    plt.title(title, fontsize=15)
    # 关闭坐标轴
    plt.axis('off')

    # 添加颜色柱 (Colorbar)
    # fraction: 颜色条占原图的比例; pad: 颜色条与图的间距
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Intensity / Confidence', rotation=-90, va="bottom")

    # 保存总图
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()


# ==========================================
# 4. 执行流程
# ==========================================
def run_pipeline(vis_path, ir_path, weight_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((448, 448)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载图像
    img_vis = cv2.imdecode(np.fromfile(vis_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    img_ir = cv2.imdecode(np.fromfile(ir_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    vis_t = transform(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)).unsqueeze(0)
    ir_t = transform(cv2.cvtColor(img_ir, cv2.COLOR_GRAY2RGB)).unsqueeze(0)

    # 模型加载
    backbone = Res2NetBackbone(weight_path).eval()
    sd_module = SpatialDecisionModule(in_channels=1024).eval()

    with torch.no_grad():
        f_vis = backbone(vis_t)
        f_ir = backbone(ir_t)
        g_vis, g_ir, conf_grad, vis_cue, sdf = sd_module(f_vis, f_ir)

    # --- 1. 独立输出五个中间过程图 (无 Colorbar) ---
    save_map_as_jet(g_vis, os.path.join(output_dir, "Step1_Visible_Gradient.png"))
    save_map_as_jet(g_ir, os.path.join(output_dir, "Step2_IR_Gradient.png"))
    save_map_as_jet(conf_grad, os.path.join(output_dir, "Step3_Gradient_Confidence.png"))
    save_map_as_jet(vis_cue, os.path.join(output_dir, "Step4_Learned_Visible_Cue.png"))
    save_map_as_jet(sdf, os.path.join(output_dir, "Step5_Spatial_Decision_Field.png"))

    # --- 2. 额外输出一张带 Colorbar 的总图 ---
    save_master_map_with_colorbar(sdf, os.path.join(output_dir, "FINAL_MASTER_SDF_with_Colorbar.png"))

    print(f"📊 任务完成！总图及中间结果已保存至: {output_dir}")


if __name__ == "__main__":
    # 配置你的路径
    W_PATH = r'F:\CVPR_2026_code_data\quwu_daima_xiugai_shanchuan_github\model\imagenet_model\res2net101_v1b_26w_4s-0812c246.pth'
    V_IN = r'F:\ACMMM2026\论文里的图优化\ReMix方案图优化\RGB_IR\hazy_RGB\image1.png'
    I_IN = r'F:\ACMMM2026\论文里的图优化\ReMix方案图优化\RGB_IR\IR\image1.png'
    OUT = r'F:\ACMMM2026\论文里的图优化\ReMix方案图优化\RGB_IR\空间决策场输出'

    run_pipeline(V_IN, I_IN, W_PATH, OUT)