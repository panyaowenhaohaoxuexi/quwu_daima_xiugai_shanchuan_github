# -*- coding: utf-8 -*-
"""
基于 OpenCV 的雾气/烟雾掩码提取脚本。
输出：mask 为 0/255 二值图；overlay 为红色半透明叠加预览。

用法示例：
python fog_mask_opencv.py --visible /path/to/visible.png --ir /path/to/infrared.png --outdir ./fog_out
"""

import argparse
from pathlib import Path
import cv2
import numpy as np


def _local_std(gray: np.ndarray, k: int = 41) -> np.ndarray:
    """计算局部标准差，用来抑制纹理较强的山体、草地、建筑等区域。"""
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (k, k))
    mean2 = cv2.blur(gray_f * gray_f, (k, k))
    return np.sqrt(np.maximum(mean2 - mean * mean, 0))


def _keep_components(mask: np.ndarray, min_area: int, keep_fn) -> np.ndarray:
    """按连通域过滤。"""
    h, w = mask.shape[:2]
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = np.zeros_like(mask)
    for i in range(1, num):
        x, y, ww, hh, area = stats[i]
        if area >= min_area and keep_fn(x, y, ww, hh, area, w, h):
            out[labels == i] = 255
    return out


def _overlay_red(img: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """生成红色半透明预览图。"""
    if img.ndim == 2:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis = img.copy()
    red = np.array([0, 0, 255], dtype=np.float32)  # OpenCV: BGR
    idx = mask > 0
    vis[idx] = ((1 - alpha) * vis[idx].astype(np.float32) + alpha * red).astype(np.uint8)
    return vis


def fog_mask_visible(img_bgr: np.ndarray, sky_cut: float = 0.32) -> np.ndarray:
    """
    可见光雾气掩码。
    规则：雾/烟通常表现为低饱和、高亮度、高暗通道、低纹理的大块区域。
    sky_cut 用来去掉图像上方天空误检；你的样图中雾气主要位于中下部和右侧。
    """
    h, w = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    _, sat, _ = cv2.split(hsv)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    min_channel = img_bgr.min(axis=2)
    std = _local_std(gray, 41)

    # 归一化得分：低饱和 + 高暗通道 + 低纹理
    score = (
        0.35 * np.clip((65 - sat) / 65, 0, 1) +
        0.50 * np.clip((min_channel - 110) / 85, 0, 1) +
        0.15 * np.clip((40 - std) / 40, 0, 1)
    )

    mask = ((score > 0.55) & (min_channel >= 115)).astype(np.uint8) * 255
    mask[: int(sky_cut * h), :] = 0

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)), iterations=2)

    # 保留面积较大、靠近右侧或中下部的雾团；可根据数据集改掉这个先验。
    mask = _keep_components(
        mask,
        min_area=2000,
        keep_fn=lambda x, y, ww, hh, area, W, H: (x + ww > 0.45 * W) and (y + hh > 0.45 * H),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35)), iterations=1)
    return mask


def fog_mask_ir(img_gray: np.ndarray) -> np.ndarray:
    """
    红外/灰度雾气掩码。
    规则：雾/烟在红外图里常呈现为平滑、较亮、低纹理的团状区域。
    你的样图雾气主要从右下方进入，因此加入右侧/下侧空间先验。
    """
    if img_gray.ndim == 3:
        img_gray = cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)

    h, w = img_gray.shape[:2]
    y, x = np.indices((h, w))
    std = _local_std(img_gray, 41)

    # 右侧/下侧空间先验；若雾气位置不同，调小这些比例或直接删掉 spatial。
    spatial = ((x > 0.65 * w) & (y > 0.46 * h)) | ((x > 0.42 * w) & (y > 0.72 * h))
    mask = ((img_gray > 95) & (std < 24) & spatial).astype(np.uint8) * 255

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (37, 37)), iterations=2)

    mask = _keep_components(
        mask,
        min_area=1200,
        keep_fn=lambda x, y, ww, hh, area, W, H: (x + ww > W - 5) or (y + hh > H - 5),
    )
    return mask


def process(visible_path: str | None, ir_path: str | None, outdir: str) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    if visible_path:
        img = cv2.imread(visible_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"无法读取可见光图像：{visible_path}")
        mask = fog_mask_visible(img)
        cv2.imwrite(str(out / "visible_fog_mask.png"), mask)
        cv2.imwrite(str(out / "visible_fog_overlay.png"), _overlay_red(img, mask))

    if ir_path:
        img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"无法读取红外图像：{ir_path}")
        mask = fog_mask_ir(img)
        cv2.imwrite(str(out / "ir_fog_mask.png"), mask)
        cv2.imwrite(str(out / "ir_fog_overlay.png"), _overlay_red(img, mask))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--visible", type=str, default=None, help="可见光图像路径")
    parser.add_argument("--ir", type=str, default=None, help="红外/灰度图像路径")
    parser.add_argument("--outdir", type=str, default="fog_out", help="输出目录")
    args = parser.parse_args()
    process(args.visible, args.ir, args.outdir)
