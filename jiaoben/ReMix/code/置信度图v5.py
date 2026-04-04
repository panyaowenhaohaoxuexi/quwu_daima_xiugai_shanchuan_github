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
    print("错误: 请安装 timm 库 (pip install timm)")


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
# 2. 空间决策场模块
# ==========================================
class SpatialDecisionModule(nn.Module):
    def __init__(self, in_channels=1024):
        super(SpatialDecisionModule, self).__init__()
        kernel_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        kernel_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        self.register_buffer('k_x', torch.FloatTensor(kernel_x).view(1, 1, 3, 3))
        self.register_buffer('k_y', torch.FloatTensor(kernel_y).view(1, 1, 3, 3))
        self.cue_conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def _get_gradient(self, x):
        x_mean = torch.mean(x, dim=1, keepdim=True)
        grad_x = F.conv2d(x_mean, self.k_x, padding=1)
        grad_y = F.conv2d(x_mean, self.k_y, padding=1)
        return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)

    def forward(self, f_vis, f_ir):
        g_vis = self._get_gradient(f_vis)
        g_ir = self._get_gradient(f_ir)
        conf_grad = torch.exp(torch.abs(g_vis - g_ir))
        conf_grad = (conf_grad - conf_grad.min()) / (conf_grad.max() - conf_grad.min() + 1e-8)
        vis_cue = self.sigmoid(self.cue_conv(f_vis))
        sdf = conf_grad * vis_cue
        return g_vis, g_ir, conf_grad, vis_cue, sdf


# ==========================================
# 3. 交互式 ROI 选择器
# ==========================================
class InteractiveROI:
    def __init__(self, base_map):
        self.base_map = base_map
        self.drawing = False
        self.rects = []
        self.ix, self.iy = -1, -1

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                img_copy = self.get_vis().copy()
                cv2.rectangle(img_copy, (self.ix, self.iy), (x, y), (255, 255, 255), 2)
                cv2.imshow('Draw-to-Boost-Heatmap', img_copy)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.rects.append(((self.ix, self.iy), (x, y)))
            self.refresh()

    def get_vis(self):
        v = (self.base_map - self.base_map.min()) / (self.base_map.max() - self.base_map.min() + 1e-8)
        vis = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_JET)
        for r in self.rects:
            cv2.rectangle(vis, r[0], r[1], (0, 255, 0), 2)
        return vis

    def refresh(self):
        cv2.imshow('Draw-to-Boost-Heatmap', self.get_vis())

    def get_final_mask(self):
        h, w = self.base_map.shape
        user_mask = np.zeros((h, w), dtype=np.float32)
        for (p1, p2) in self.rects:
            cv2.rectangle(user_mask, p1, p2, (1.0), -1)
        # 高斯平滑
        user_mask = cv2.GaussianBlur(user_mask, (101, 101), 0)
        return user_mask


# ==========================================
# 4. 执行流水线
# ==========================================
def run_interactive_system(config):
    # --- 自动检查路径是否存在 ---
    for key in ['weight', 'vis_in', 'ir_in']:
        if not os.path.exists(config[key]):
            raise FileNotFoundError(f"❌ 找不到文件: {config[key]}\n请检查主函数中的路径配置！")

    os.makedirs(config['out_dir'], exist_ok=True)

    # 数据转换
    transform = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((448, 448)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 读取图像 (支持中文路径)
    img_vis_raw = cv2.imdecode(np.fromfile(config['vis_in'], dtype=np.uint8), 1)
    img_ir_raw = cv2.imdecode(np.fromfile(config['ir_in'], dtype=np.uint8), 0)

    vis_t = transform(cv2.cvtColor(img_vis_raw, cv2.COLOR_BGR2RGB)).unsqueeze(0)
    ir_t = transform(cv2.cvtColor(img_ir_raw, cv2.COLOR_GRAY2RGB)).unsqueeze(0)

    # 模型推理
    backbone = Res2NetBackbone(config['weight']).eval()
    sd_module = SpatialDecisionModule().eval()
    with torch.no_grad():
        feat_vis, feat_ir = backbone(vis_t), backbone(ir_t)
        g_vis, g_ir, conf_grad, vis_cue, sdf_base = sd_module(feat_vis, feat_ir)

    # 交互调整
    sdf_np = F.interpolate(sdf_base, size=(448, 448), mode='bilinear').squeeze().cpu().numpy()
    roi_selector = InteractiveROI(sdf_np)
    cv2.namedWindow('Draw-to-Boost-Heatmap')
    cv2.setMouseCallback('Draw-to-Boost-Heatmap', roi_selector.mouse_callback)
    roi_selector.refresh()

    print("\n[操作指引]")
    print("1. 在弹出窗口中用鼠标【划框】来增强特定区域的热力。")
    print("2. 按下【R】键重置，按下【Enter】完成。")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13: break
        if key == ord('r'):
            roi_selector.rects = [];
            roi_selector.refresh()

    user_guidance = roi_selector.get_final_mask()
    cv2.destroyAllWindows()

    # 合并结果
    final_sdf = sdf_np + (user_guidance * 1.5)
    final_sdf = (final_sdf - final_sdf.min()) / (final_sdf.max() - final_sdf.min() + 1e-8)

    # 保存总图 (带 Colorbar)
    plt.figure(figsize=(10, 8))
    im = plt.imshow(final_sdf, cmap='jet')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("Interactive Spatial Decision Field")
    plt.axis('off')
    plt.savefig(os.path.join(config['out_dir'], "FINAL_SDF_COLORBAR.png"), dpi=300)

    # 保存中间过程图
    plt.imsave(os.path.join(config['out_dir'], "1_Vis_Gradient.png"), g_vis.squeeze().cpu().numpy(), cmap='jet')
    plt.imsave(os.path.join(config['out_dir'], "4_Learned_Cue.png"), vis_cue.squeeze().cpu().numpy(), cmap='jet')
    plt.imsave(os.path.join(config['out_dir'], "5_Interactive_SDF.png"), final_sdf, cmap='jet')

    print(f"✅ 处理成功！文件保存在: {config['out_dir']}")


# ==========================================
# 5. 路径配置 (请务必在此填写你的真实路径)
# ==========================================
if __name__ == "__main__":
    # ⚠️ 请根据你的硬盘文件夹情况修改以下字符串
    CONFIG = {
        # 权重文件路径
        'weight': r'F:\CVPR_2026_code_data\quwu_daima_xiugai_shanchuan_github\model\imagenet_model\res2net101_v1b_26w_4s-0812c246.pth',

        # 可见光输入路径
        'vis_in': r'F:\ACMMM2026\论文里的图优化\ReMix方案图优化\RGB_IR\hazy_RGB\image1.png',

        # 红外输入路径
        'ir_in': r'F:\ACMMM2026\论文里的图优化\ReMix方案图优化\RGB_IR\IR\image1.png',

        # 输出目录
        'out_dir': r'F:\ACMMM2026\论文里的图优化\ReMix方案图优化\RGB_IR\交互式结果'
    }

    run_interactive_system(CONFIG)