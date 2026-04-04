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
# 2. 可调控空间决策场模块 (Adjustable SDM)
# ==========================================
class SpatialDecisionModule(nn.Module):
    def __init__(self, in_channels=1024):
        super(SpatialDecisionModule, self).__init__()
        # 梯度算子
        kernel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        kernel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        self.register_buffer('k_x', torch.FloatTensor(kernel_x).view(1, 1, 3, 3))
        self.register_buffer('k_y', torch.FloatTensor(kernel_y).view(1, 1, 3, 3))

        # 学习路径
        self.cue_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def _get_gradient(self, x):
        x_mean = torch.mean(x, dim=1, keepdim=True)
        grad_x = F.conv2d(x_mean, self.k_x, padding=1)
        grad_y = F.conv2d(x_mean, self.k_y, padding=1)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

    def forward(self, f_vis, f_ir, alpha=1.0, beta=1.0, gamma=1.0):
        """
        参数说明：
        :param alpha: 梯度增强因子。增大它会使梯度差异更显著（红区更红）。
        :param beta:  线索增强因子。增大它会增强可见光特征的权重。
        :param gamma: 全局对比度因子 (Power Law Transform)。
                      >1: 颜色向两极分化（红更红，蓝更蓝）；
                      <1: 颜色分布更平滑。
        """
        # A. 物理梯度置信度 (归一化到 0-1 方便幂次运算)
        g_vis = self._get_gradient(f_vis)
        g_ir = self._get_gradient(f_ir)
        conf_grad = torch.exp(torch.abs(g_vis - g_ir))
        conf_grad = (conf_grad - conf_grad.min()) / (conf_grad.max() - conf_grad.min() + 1e-8)
        conf_grad = torch.pow(conf_grad, alpha)  # 手动调节梯度分布

        # B. 语义学习路径 (Learned Visible Cue)
        vis_cue = self.sigmoid(self.cue_conv(f_vis))
        vis_cue = torch.pow(vis_cue, beta)  # 手动调节线索分布

        # C. 融合并应用全局对比度调节
        sdf = torch.pow(conf_grad * vis_cue, gamma)

        return g_vis, g_ir, conf_grad, vis_cue, sdf


# ==========================================
# 3. 可视化与保存
# ==========================================
def normalize_and_save(tensor, path, cmap='jet', has_colorbar=False, title=""):
    img = F.interpolate(tensor, size=(448, 448), mode='bilinear').squeeze().cpu().numpy()
    img_norm = (img - img.min()) / (img.max() - img.min() + 1e-8)

    if has_colorbar:
        plt.figure(figsize=(10, 8))
        im = plt.imshow(img_norm, cmap=cmap)
        plt.title(title)
        plt.axis('off')
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.savefig(path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.imsave(path, img_norm, cmap=cmap)


def run_pipeline(vis_path, ir_path, weight_path, output_dir, params):
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
        # 传入手动调节参数
        g_vis, g_ir, conf_grad, vis_cue, sdf = sd_module(f_vis, f_ir,
                                                         alpha=params['alpha'],
                                                         beta=params['beta'],
                                                         gamma=params['gamma'])

    # 输出图像
    normalize_and_save(g_vis, os.path.join(output_dir, "1_Visible_Gradient.png"))
    normalize_and_save(g_ir, os.path.join(output_dir, "2_IR_Gradient.png"))
    normalize_and_save(conf_grad, os.path.join(output_dir, "3_Adjusted_Gradient_Conf.png"))
    normalize_and_save(vis_cue, os.path.join(output_dir, "4_Adjusted_Visible_Cue.png"))
    normalize_and_save(sdf, os.path.join(output_dir, "5_Final_SDF.png"))

    # 输出带 Colorbar 的总图
    normalize_and_save(sdf, os.path.join(output_dir, "MASTER_SDF_WITH_COLORBAR.png"),
                       has_colorbar=True, title=f"SDF (a={params['alpha']}, b={params['beta']}, g={params['gamma']})")

    print(f"🎨 调节完成！当前参数: {params}")


# ==========================================
# 4. 手动调节面板
# ==========================================
if __name__ == "__main__":
    # --- 在这里修改参数来调整你的颜色分布 ---
    MY_PARAMS = {
        'alpha': 1.5,  # 提高这个值，会让“差异显著”的地方颜色更红
        'beta': 1.0,  # 提高这个值，会更看重可见光提取的线索
        'gamma': 2.0  # 提高这个值，会增强整体对比度，清除掉背景的“浅蓝色”噪声
    }

    W_PATH = r'F:\CVPR_2026_code_data\quwu_daima_xiugai_shanchuan_github\model\imagenet_model\res2net101_v1b_26w_4s-0812c246.pth'
    V_IN = r'F:\ACMMM2026\论文里的图优化\ReMix方案图优化\RGB_IR\hazy_RGB\image1.png'
    I_IN = r'F:\ACMMM2026\论文里的图优化\ReMix方案图优化\RGB_IR\IR\image1.png'
    OUT = r'F:\ACMMM2026\论文里的图优化\ReMix方案图优化\RGB_IR\空间决策场输出'

    run_pipeline(V_IN, I_IN, W_PATH, OUT, MY_PARAMS)