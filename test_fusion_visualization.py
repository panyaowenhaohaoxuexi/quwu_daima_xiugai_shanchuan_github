"""融合中间特征层结构对比可视化。

核心对比：A（纯可见光 RGB 外观特征）vs fusion_candidate（TIR 结构增强后的融合特征）
目标：证明融合方案增强了可见光融合区域的结构信息。

方法：
  1. 逐通道梯度能量对比（16 通道柱状图）
  2. 特征结构图对比（平均梯度幅值 → 空间热力图）
  3. PCA 降维可视化 + 边缘检测
  4. 增量最大的通道单独展示
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as TF

from data.data_loader import load_tir_as_float_tensor
from utils.checkpoint import (
    build_model_from_config,
    load_strict_v2_state_dict,
    preflight_eval_checkpoint,
    tir_normalization_config,
)


# ============================================================
# 测试路径
# ============================================================
DEFAULT_CHECKPOINT = r"Teacher_Train/source_best.pt"
DEFAULT_HAZY = r"F:/1_paper_pan/1_Dehaze_Paper/2_Dataset/1_main_benchmark/1_FLIR/test/hazy/3_dense/FLIR_04884.jpg"
DEFAULT_TIR = r"F:/1_paper_pan/1_Dehaze_Paper/2_Dataset/1_main_benchmark/1_FLIR/test/ir/FLIR_04884.jpg"
DEFAULT_OUTPUT = r"F:/1_paper_pan/1_Dehaze_Paper/3_conference/7_TIP/2-论文里的图/中间可视化/ReMix中间过程可视化/融合流"


def build_parser():
    parser = argparse.ArgumentParser("fusion-feature-structure-vis")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--hazy", default=DEFAULT_HAZY)
    parser.add_argument("--tir", default=DEFAULT_TIR)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--format", default="png")
    return parser


# ============================================================
# 专业伪彩色映射 — 蓝色系，清爽明亮
# ============================================================

def _blues_colormap(arr):
    """白→浅蓝→天蓝→深蓝，干净清爽。"""
    arr = np.clip(arr, 0, 1)
    r = np.interp(arr, [0, 0.3, 0.6, 0.85, 1.0], [0.97, 0.78, 0.45, 0.15, 0.03])
    g = np.interp(arr, [0, 0.3, 0.6, 0.85, 1.0], [0.98, 0.85, 0.68, 0.45, 0.20])
    b = np.interp(arr, [0, 0.3, 0.6, 0.85, 1.0], [0.97, 0.94, 0.90, 0.80, 0.55])
    return np.stack([r, g, b], axis=-1)


def _warm_colormap(arr):
    """暖色系：浅黄→橙→珊瑚红，柔和温暖不恐怖。"""
    arr = np.clip(arr, 0, 1)
    r = np.interp(arr, [0, 0.3, 0.6, 0.85, 1.0], [0.99, 0.99, 0.95, 0.85, 0.60])
    g = np.interp(arr, [0, 0.3, 0.6, 0.85, 1.0], [0.97, 0.82, 0.55, 0.30, 0.10])
    b = np.interp(arr, [0, 0.3, 0.6, 0.85, 1.0], [0.85, 0.55, 0.35, 0.20, 0.12])
    return np.stack([r, g, b], axis=-1)


def _coolwarm_colormap(arr):
    """浅蓝→白→浅橙，diverging 柔和版。"""
    arr = np.clip(arr, -1, 1)
    x = (arr + 1) / 2  # [0, 1]
    r = np.interp(x, [0, 0.35, 0.5, 0.65, 1.0], [0.35, 0.68, 0.95, 0.98, 0.82])
    g = np.interp(x, [0, 0.35, 0.5, 0.65, 1.0], [0.55, 0.82, 0.95, 0.72, 0.38])
    b = np.interp(x, [0, 0.35, 0.5, 0.65, 1.0], [0.82, 0.95, 0.95, 0.60, 0.28])
    return np.stack([r, g, b], axis=-1)


# 兼容旧接口
_apply_hot_colormap = _blues_colormap
_apply_blue_red_colormap = _coolwarm_colormap


def _save_tensor(tensor, path, *, cmap=None, remark=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if cmap is not None and tensor.ndim == 3 and tensor.shape[0] == 1:
        arr = tensor.detach().float().cpu()[0].clamp(0, 1).numpy()
        if cmap == "hot" or cmap == "viridis":
            colored = _blues_colormap(arr)
            img = Image.fromarray((colored * 255).astype(np.uint8))
        elif cmap == "magma":
            colored = _warm_colormap(arr)
            img = Image.fromarray((colored * 255).astype(np.uint8))
        elif cmap == "bwr":
            arr_centered = arr * 2 - 1
            colored = _coolwarm_colormap(arr_centered)
            img = Image.fromarray((colored * 255).astype(np.uint8))
        else:
            img = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
        img.save(path)
    else:
        if tensor.ndim == 4:
            t = tensor[0]
        else:
            t = tensor
        TF.to_pil_image(t.cpu().clamp(0, 1)).save(path)
    print(f"  已保存: {path}")


def _add_label(img_pil, text, position="top-left", font_size=16):
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("simhei.ttf", font_size)
    except OSError:
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except OSError:
            font = ImageFont.load_default()
    w, h = img_pil.size
    if position == "top-left":
        xy = (8, 8)
    elif position == "bottom-left":
        bbox = draw.textbbox((0, 0), text, font=font)
        xy = (8, h - bbox[3] - 12)
    else:
        bbox = draw.textbbox((0, 0), text, font=font)
        xy = (w - bbox[2] - 8, 8)
    draw.text((xy[0] + 1, xy[1] + 1), text, font=font, fill="black")
    draw.text(xy, text, font=font, fill="white")
    return img_pil


# ============================================================
# 特征结构分析
# ============================================================

def _feature_gradient_magnitude(feature):
    """计算 [C,H,W] 特征每个通道的 Sobel 梯度幅值，返回 [C,H,W]."""
    if feature.ndim == 4:
        feature = feature[0]
    C, H, W = feature.shape
    feat = feature.float()
    # 用 3×3 Sobel 核
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=feat.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=feat.device).view(1, 1, 3, 3)
    # 逐通道卷积
    feat_4d = feat.unsqueeze(1)  # [C, 1, H, W]
    gx = F.conv2d(F.pad(feat_4d, (1, 1, 1, 1), mode='replicate'), kx)
    gy = F.conv2d(F.pad(feat_4d, (1, 1, 1, 1), mode='replicate'), ky)
    return torch.sqrt(gx.square() + gy.square() + 1e-8).squeeze(1)  # [C, H, W]


def _per_channel_edge_energy(feature):
    """返回每个通道的梯度能量（标量列表）。"""
    grad = _feature_gradient_magnitude(feature)  # [C, H, W]
    return grad.reshape(grad.shape[0], -1).mean(dim=1)  # [C]


def _mean_structure_map(feature):
    """特征的平均结构图：逐通道梯度幅值的均值，归一化到 [0,1]."""
    grad = _feature_gradient_magnitude(feature)  # [C, H, W]
    structure = grad.mean(dim=0, keepdim=True)    # [1, H, W]
    structure = structure / structure.max().clamp_min(1e-8)
    return structure


def _pca_rgb(feature):
    """PCA 降维到 3 通道 RGB."""
    if feature.ndim == 4:
        feat = feature[0]
    else:
        feat = feature
    C, H, W = feat.shape
    X = feat.reshape(C, -1).T  # [H*W, C]
    X_mean = X.mean(dim=0, keepdim=True)
    X_centered = X - X_mean
    U, S, V = torch.linalg.svd(X_centered, full_matrices=False)
    pca_3 = X_centered @ V.T[:, :3]  # [H*W, 3]
    p_min = pca_3.min(dim=0).values
    p_max = pca_3.max(dim=0).values
    pca_norm = (pca_3 - p_min) / (p_max - p_min).clamp_min(1e-8)
    return pca_norm.T.reshape(1, 3, H, W)


def _pca_edge_map(feature):
    """对 PCA 降维后的 RGB 图做边缘检测。"""
    rgb = _pca_rgb(feature)  # [1, 3, H, W]
    pil_img = TF.to_pil_image(rgb[0].cpu().clamp(0, 1))
    gray = pil_img.convert("L")
    from PIL import ImageFilter
    edge = gray.filter(ImageFilter.FIND_EDGES)
    edge_t = TF.pil_to_tensor(edge).float().div_(255)
    if edge_t.max() > 0:
        edge_t = edge_t / edge_t.max()
    return edge_t.unsqueeze(0)  # [1, 1, H, W]


# ============================================================
# 柱状图渲染（纯 numpy/PIL，不依赖 matplotlib）
# ============================================================

def _draw_bar_chart(values_a, values_b, labels, title, xlabel, path, figsize=(1200, 500)):
    """绘制双组柱状图：A（蓝色）vs fusion_candidate（红色）。"""
    n = len(values_a)
    w_bar = int(figsize[0] / (n * 3))
    margin_left, margin_right, margin_top, margin_bottom = 100, 40, 50, 80
    plot_w = figsize[0] - margin_left - margin_right
    plot_h = figsize[1] - margin_top - margin_bottom

    img = Image.new("RGB", figsize, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("simhei.ttf", 18)
        font_axis = ImageFont.truetype("simhei.ttf", 12)
        font_tick = ImageFont.truetype("simhei.ttf", 10)
    except OSError:
        font_title = font_axis = font_tick = ImageFont.load_default()

    max_val = max(max(values_a), max(values_b)) * 1.15
    if max_val == 0:
        max_val = 1

    def _val_to_y(v):
        return margin_top + plot_h - int(v / max_val * plot_h)

    # Y 轴刻度
    for i in range(5):
        frac = i / 4.0
        y = _val_to_y(frac * max_val)
        label = f"{frac * max_val:.3f}"
        draw.line([(margin_left - 5, y), (margin_left, y)], fill=(180, 180, 180), width=1)
        draw.text((margin_left - 50, y - 7), label, fill=(80, 80, 80), font=font_tick)
    draw.line([(margin_left, margin_top), (margin_left, margin_top + plot_h)], fill=(0, 0, 0), width=1)
    draw.line([(margin_left, margin_top + plot_h), (margin_left + plot_w, margin_top + plot_h)], fill=(0, 0, 0), width=1)

    # 柱子 — 蓝色系
    COLOR_A = (150, 190, 230)       # light steel blue
    COLOR_FUSION = (50, 120, 200)   # medium blue
    for i in range(n):
        x_center = margin_left + int((i + 0.5) / n * plot_w)
        # A 柱（steel blue）
        draw.rectangle([x_center - w_bar, _val_to_y(values_a[i]), x_center, margin_top + plot_h],
                       fill=COLOR_A)
        # fusion 柱（coral）
        draw.rectangle([x_center, _val_to_y(values_b[i]), x_center + w_bar, margin_top + plot_h],
                       fill=COLOR_FUSION)
        # 通道标签
        draw.text((x_center - w_bar // 2, margin_top + plot_h + 5), labels[i],
                  fill=(60, 60, 60), font=font_tick, anchor="mt")

    # 图例
    draw.rectangle([margin_left + 10, margin_top + 8, margin_left + 28, margin_top + 20], fill=COLOR_A)
    draw.text((margin_left + 34, margin_top + 6), "A (可见光特征)", fill=(60, 60, 60), font=font_axis)
    draw.rectangle([margin_left + 180, margin_top + 8, margin_left + 198, margin_top + 20], fill=COLOR_FUSION)
    draw.text((margin_left + 204, margin_top + 6), "fusion_candidate (融合增强)", fill=(60, 60, 60), font=font_axis)

    # 标题
    draw.text((figsize[0] // 2, margin_top - 30), title, fill=(30, 30, 30), font=font_title, anchor="mt")
    draw.text((figsize[0] // 2, figsize[1] - 15), xlabel, fill=(80, 80, 80), font=font_axis, anchor="mm")

    img.save(path)
    print(f"  已保存柱状图: {path}")
    return img


# ============================================================
# 主流程
# ============================================================

def main(argv=None):
    args = build_parser().parse_args(argv)

    # --- 1. 加载模型 ---
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    preflight = preflight_eval_checkpoint(checkpoint, ema_model="teacher")
    config = preflight["config"]
    print(f"Checkpoint: {preflight['stage']} | state: {preflight['state_key']}")

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )
    model = build_model_from_config(config).to(device)
    load_strict_v2_state_dict(model, preflight["state_dict"], label=preflight["state_label"])
    model.eval()
    print(f"模型已加载到 {device}")

    # --- 2. 加载图像 ---
    with Image.open(args.hazy) as img:
        hazy = TF.pil_to_tensor(img.convert("RGB")).float().div_(255).unsqueeze(0)
    tir_config = tir_normalization_config(config)
    tir = load_tir_as_float_tensor(args.tir, tir_config).unsqueeze(0)
    if tir.shape[-2:] != hazy.shape[-2:]:
        tir = F.interpolate(tir, size=hazy.shape[-2:], mode="bilinear", align_corners=False)
    print(f"Hazy: {tuple(hazy.shape)}  |  TIR: {tuple(tir.shape)}")

    # --- 3. 推理，捕获中间特征 ---
    route_temperature = config.get("route_tau_end", 0.2)
    with torch.inference_mode():
        context = model.encode_context(hazy.to(device), tir.to(device), route_temperature=route_temperature)
        output = model.decode_with_route(
            context, route_mode="hard",
            capture_fusion_intermediates=True,
        )

    intermediates = output["fusion_intermediates"]
    A = intermediates["A"].cpu()                    # [1, 16, H/2, W/2]
    fusion_cand = intermediates["fusion_candidate"].cpu()
    delta = intermediates["delta"].cpu()
    route_hard = output["route_hard"].cpu()
    density_map = output["density_map"].cpu()
    pred_clear = output["pred_clear"].cpu()
    tir_rgb = tir[:, :1].repeat(1, 3, 1, 1) if tir.shape[1] == 1 else tir

    C = A.shape[1]
    print(f"h2 特征通道数: {C}, 空间尺寸: {tuple(A.shape[-2:])}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.hazy).stem
    fmt = str(args.format).lstrip(".")

    # ================================================================
    # A. 逐通道梯度能量对比
    # ================================================================
    print("\n=== A. 逐通道梯度能量对比 ===")
    edge_A = _per_channel_edge_energy(A)           # [16]
    edge_fusion = _per_channel_edge_energy(fusion_cand)  # [16]
    edge_delta = _per_channel_edge_energy(delta)   # [16]

    channel_labels = [f"ch{i}" for i in range(C)]
    _draw_bar_chart(
        edge_A.tolist(), edge_fusion.tolist(), channel_labels,
        title=f"Per-Channel Gradient Energy: A vs fusion_candidate ({stem})",
        xlabel=f"Channel (共 {C} 通道, h2 尺度)",
        path=out_dir / f"{stem}_channel_edge_energy.{fmt}",
    )

    # 打印通道级增强统计
    enhancement_ratio = ((edge_fusion - edge_A) / edge_A.clamp_min(1e-8)).tolist()
    top_channels = sorted(range(C), key=lambda i: enhancement_ratio[i], reverse=True)
    print(f"  结构增强最多的前5通道: {top_channels[:5]}")
    print(f"  增强比例: {[f'{enhancement_ratio[i]:.2%}' for i in top_channels[:5]]}")
    print(f"  平均增强: {np.mean(enhancement_ratio):.2%}")

    # ================================================================
    # B. 特征图 PCA 可视化（中间层特征本身）
    # ================================================================
    print("\n=== B. 特征图 PCA 可视化 ===")
    pca_A = _pca_rgb(A)               # 可见光特征本身
    pca_fusion = _pca_rgb(fusion_cand) # 融合增强特征本身
    pca_delta = _pca_rgb(delta * 5.0)  # delta 信号弱，放大

    _save_tensor(pca_A, out_dir / f"{stem}_feature_A_pca.{fmt}",
                 remark="可见光特征 A 的 PCA 可视化（中间层特征图）")
    _save_tensor(pca_fusion, out_dir / f"{stem}_feature_fusion_pca.{fmt}",
                 remark="融合增强特征 fusion_candidate 的 PCA 可视化")

    # ================================================================
    # C. 梯度结构图（展示"结构信息"的量化证据）
    # ================================================================
    print("\n=== C. 梯度结构图 ===")
    struct_A = _mean_structure_map(A)
    struct_fusion = _mean_structure_map(fusion_cand)
    struct_delta_map = _mean_structure_map(delta)

    _save_tensor(struct_A, out_dir / f"{stem}_gradient_A.{fmt}", cmap="hot",
                 remark="A 的逐通道梯度幅值均值（结构强度）")
    _save_tensor(struct_fusion, out_dir / f"{stem}_gradient_fusion.{fmt}", cmap="hot",
                 remark="fusion_candidate 的逐通道梯度幅值均值")
    struct_diff = (struct_fusion - struct_A).clamp_min(0)
    _save_tensor(struct_diff, out_dir / f"{stem}_gradient_enhancement.{fmt}", cmap="hot",
                 remark="梯度增强区域：fusion 比 A 结构更强的地方")

    # ================================================================
    # D. 核心面板：特征层结构对比（2行 × 4列）
    # ================================================================
    print("\n=== D. 核心面板 ===")

    # 第1行：输入 + 特征图 PCA
    # 第2行：梯度结构 + 路由图
    route_vis = 1.0 - route_hard.float()  # 白=融合区域

    row1 = [hazy, pca_A, pca_fusion, pca_delta]
    row1_labels = [
        "(a) Hazy Input",
        "(b) A (可见光特征 PCA)",
        "(c) fusion_candidate (融合特征 PCA)",
        "(d) delta×5 (TIR贡献 PCA)",
    ]
    row2 = [struct_A, struct_fusion, struct_diff, route_vis]
    row2_labels = [
        "(e) A 梯度结构 (可见光)",
        "(f) fusion 梯度结构 (融合增强)",
        "(g) 结构增强区域 (f-A>0)",
        "(h) Route Map (白=融合区域)",
    ]

    all_tensors = row1 + row2
    all_labels = row1_labels + row2_labels

    panel_imgs = []
    for t, label in zip(all_tensors, all_labels):
        if t.shape[0] == 1:
            pil_img = TF.to_pil_image(t[0].cpu().clamp(0, 1))
        else:
            pil_img = TF.to_pil_image(t.cpu().clamp(0, 1))
        panel_imgs.append(_add_label(pil_img, label, font_size=13))

    w, h = panel_imgs[0].size
    ncols = 4
    panel = Image.new("RGB", (w * ncols, h * 2), color=(255, 255, 255))
    for idx, img in enumerate(panel_imgs):
        row, col = divmod(idx, ncols)
        panel.paste(img, (col * w, row * h))
    panel.save(out_dir / f"{stem}_feature_structure_panel.{fmt}")
    print(f"  已保存核心面板: {out_dir / f'{stem}_feature_structure_panel.{fmt}'}")

    # ================================================================
    # E. 单通道细节面板（增强最大的 4 个通道）
    # ================================================================
    print("\n=== E. 单通道细节面板 ===")
    top4 = top_channels[:4]
    ncols_ch = 3  # A通道, fusion通道, 差异

    # 对每个通道做归一化
    channel_tiles = []
    for ch_idx in top4:
        a_ch = A[0, ch_idx:ch_idx+1]           # [1, H, W]
        f_ch = fusion_cand[0, ch_idx:ch_idx+1]
        # 统一用两者最大值归一化
        vmax = max(a_ch.max().item(), f_ch.max().item())
        a_norm = a_ch / vmax
        f_norm = f_ch / vmax
        diff_ch = (f_norm - a_norm).clamp_min(0)

        enhancement_pct = enhancement_ratio[ch_idx]
        a_label = f"ch{ch_idx} A (vis)"
        f_label = f"ch{ch_idx} fusion (+{enhancement_pct:.0%})"
        d_label = f"ch{ch_idx} diff"

        for t, label in [(a_norm, a_label), (f_norm, f_label), (diff_ch, d_label)]:
            pil = TF.to_pil_image(t.cpu().clamp(0, 1))
            channel_tiles.append(_add_label(pil, label, font_size=11))

    # 拼接: 4行 × 3列
    w_ch, h_ch = channel_tiles[0].size
    ch_panel = Image.new("RGB", (w_ch * 3, h_ch * 4), color=(255, 255, 255))
    for idx, img in enumerate(channel_tiles):
        row, col = divmod(idx, 3)
        ch_panel.paste(img, (col * w_ch, row * h_ch))
    ch_panel.save(out_dir / f"{stem}_top4_channels.{fmt}")
    print(f"  已保存通道面板: {out_dir / f'{stem}_top4_channels.{fmt}'}")

    # ================================================================
    # F. 汇总统计
    # ================================================================
    print(f"\n{'='*55}")
    print(f"特征层结构对比分析 — {stem}")
    print(f"{'='*55}")
    print(f"特征通道数: {C}  (h2 尺度)")
    print(f"特征空间尺寸: {tuple(A.shape[-2:])}")
    print()
    print(f"--- 逐通道梯度能量 ---")
    print(f"  A (可见光) 均值:           {edge_A.mean().item():.6f}")
    print(f"  fusion_candidate (融合) 均值: {edge_fusion.mean().item():.6f}")
    print(f"  delta (TIR贡献) 均值:        {edge_delta.mean().item():.6f}")
    print(f"  平均增强比例:                {np.mean(enhancement_ratio):.2%}")
    print(f"  增强最大的通道: ch{top_channels[0]} (+{enhancement_ratio[top_channels[0]]:.1%}), "
          f"ch{top_channels[1]} (+{enhancement_ratio[top_channels[1]]:.1%}), "
          f"ch{top_channels[2]} (+{enhancement_ratio[top_channels[2]]:.1%})")
    print()
    print(f"--- 结构图统计 ---")
    print(f"  A 结构图均值:            {struct_A.mean().item():.4f}")
    print(f"  fusion 结构图均值:        {struct_fusion.mean().item():.4f}")
    print(f"  结构增强区域占比 (>0):     {(struct_diff > 0.01).float().mean().item():.2%}")
    print()
    print(f"全部输出已保存到: {out_dir}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
