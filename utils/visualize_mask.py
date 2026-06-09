"""
utils/visualize_mask.py

训练过程中的掩码可视化工具。
每个 epoch 结束时调用 visualize_epoch_mask()，
保存一张包含 4 列的对比图：
  列1: 可见光雾图（反归一化到 [0,1]）
  列2: 红外图（归一化到 [0,1]）
  列3: HAPM 输出的软密度图 M_vis（连续值）
  列4: 二值化后的 haze_mask（0/1）

每个 epoch 保存到 {save_dir}/mask_vis/epoch_{epoch:03d}.png
固定取前 N 张样本，保证每轮看的是同一批图像，便于横向比较。
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')   # 非交互式后端，服务器上安全
import matplotlib.pyplot as plt


# CLIP 归一化参数（与 model/Teacher.py 中保持一致）
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073]).view(1, 3, 1, 1)
_CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def _to_numpy_rgb(tensor_chw):
    """(C,H,W) float tensor → (H,W,3) uint8 numpy，clip 到 [0,1]"""
    img = tensor_chw.detach().cpu().float()
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def _denorm_vis(vis_batch, device):
    """
    把 CLIP 归一化的可见光 batch 还原到 [0,1]。
    vis_batch: (B, 3, H, W)
    """
    mean = _CLIP_MEAN.to(device)
    std  = _CLIP_STD.to(device)
    return (vis_batch * std + mean).clamp(0, 1)


def _norm_ir(ir_batch):
    """
    红外图归一化到 [0,1] 用于显示。
    取单张 batch 的 min/max 归一化。
    ir_batch: (B, 3, H, W) 或 (B, 1, H, W)
    """
    b = ir_batch.detach().cpu().float()
    # 对每张图独立归一化
    out = []
    for i in range(b.shape[0]):
        img = b[i]
        mn, mx = img.min(), img.max()
        img = (img - mn) / (mx - mn + 1e-6)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        out.append(img)
    return torch.stack(out)   # (B, 3, H, W)


@torch.no_grad()
def visualize_epoch_mask(
    model,
    vis_batch,
    ir_batch,
    epoch,
    save_dir,
    n_samples=4,
    device='cpu'
):
    """
    可视化当前 epoch 的 haze_mask。

    参数：
        model      : teacher 网络（model/Teacher.py 的 VIFNetInconsistencyTeacher）
        vis_batch  : (B, 3, H, W) CLIP 归一化的可见光雾图，取自真实数据
        ir_batch   : (B, 3, H, W) 红外图
        epoch      : 当前 epoch 编号（用于文件命名）
        save_dir   : 输出根目录（通常是 opt.saved_data_dir）
        n_samples  : 每张图显示几个样本（行数）
        device     : 'cuda' 或 'cpu'
    """
    # 确定实际样本数
    n = min(n_samples, vis_batch.shape[0])
    vis = vis_batch[:n].to(device)
    ir  = ir_batch[:n].to(device)

    # 切换到 eval 模式，推理结束后恢复
    training_before = model.training
    model.eval()

    # ---- 用 hook 捕获 HDE 的软密度图 M_vis ----
    # HDE 的输出是 M_vis（连续值），之后才被 Otsu 二值化为 haze_mask
    # 我们同时想看软图和硬图，所以 hook HDE 的输出
    _captured = {}

    def _hde_hook(module, inp, out):
        _captured['M_vis'] = out.detach().cpu()   # (B, 1, H, W)

    # 处理 DataParallel 包装
    _model = model.module if hasattr(model, 'module') else model
    _hook = _model.hde.register_forward_hook(_hde_hook)

    try:
        # 正常前向（内部会自动生成 haze_mask）
        _ = model(vis, ir)
    except Exception as e:
        print(f"[visualize_mask] 前向推理失败: {e}")
        _hook.remove()
        if training_before:
            model.train()
        return
    finally:
        _hook.remove()

    # 恢复训练模式
    if training_before:
        model.train()

    if 'M_vis' not in _captured:
        print("[visualize_mask] 未捕获到 HDE 输出，跳过可视化。")
        return

    M_vis = _captured['M_vis'][:n]   # (n, 1, H, W)

    # 从 M_vis 重新计算 haze_mask（与 model/Teacher.py 中逻辑一致）
    # 简单用全图 0.5 阈值得到二值图（仅用于可视化，不影响训练）
    # 也可以用 Otsu，但 cpu 上直接用中位数代替更简单
    thresholds = M_vis.flatten(1).median(dim=1).values   # (n,)
    haze_mask_hard = torch.zeros_like(M_vis)
    for i in range(n):
        haze_mask_hard[i] = (M_vis[i] >= thresholds[i]).float()

    # 反归一化可见光
    vis_01   = _denorm_vis(vis, device).cpu()   # (n, 3, H, W)
    ir_01    = _norm_ir(ir)                      # (n, 3, H, W)

    # ---- 绘图 ----
    fig, axes = plt.subplots(
        nrows=n, ncols=4,
        figsize=(16, 4 * n),
        squeeze=False
    )
    col_titles = ['可见光雾图', '红外图', 'HAPM 软密度图 M_vis', '二值掩码 haze_mask']

    for row in range(n):
        # 列1：可见光
        axes[row, 0].imshow(_to_numpy_rgb(vis_01[row]))
        axes[row, 0].axis('off')

        # 列2：红外
        axes[row, 1].imshow(_to_numpy_rgb(ir_01[row]))
        axes[row, 1].axis('off')

        # 列3：软密度图（伪彩色）
        m_soft = M_vis[row, 0].numpy()   # (H, W)
        im3 = axes[row, 2].imshow(m_soft, cmap='hot', vmin=0, vmax=1)
        axes[row, 2].axis('off')
        plt.colorbar(im3, ax=axes[row, 2], fraction=0.046, pad=0.04)

        # 列4：二值掩码
        m_hard = haze_mask_hard[row, 0].numpy()   # (H, W)
        axes[row, 3].imshow(m_hard, cmap='gray', vmin=0, vmax=1)
        axes[row, 3].axis('off')

        # 行标签
        axes[row, 0].set_ylabel(f'Sample {row+1}', fontsize=10)

    # 列标题
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight='bold')

    fig.suptitle(f'Epoch {epoch}  —  Haze Mask 可视化', fontsize=13, y=1.01)
    plt.tight_layout()

    # 保存
    out_dir = os.path.join(save_dir, 'mask_vis')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'epoch_{epoch:03d}.png')
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

    print(f"\n[mask_vis] Epoch {epoch} 掩码已保存 → {out_path}")
