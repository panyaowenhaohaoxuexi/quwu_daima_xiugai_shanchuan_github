#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def haze_image(J: np.ndarray, t: float, A: np.ndarray) -> np.ndarray:
    """
    大气散射模型: I = J * t + A * (1 - t)
    J: [H,W,3] in [0,1]  原图
    t: 标量传输率(越小雾越浓)  e.g. 0.3~0.95
    A: [3,] 大气光(每通道)     e.g. 0.7~1.0
    """
    t = float(t)
    A = A.reshape(1, 1, 3)          # [1,1,3]
    I = J * t + A * (1.0 - t)
    I = np.clip(I, 0.0, 1.0)
    return I

def process_one(in_path: Path, out_path: Path, t_min: float, t_max: float,
                A_min: float, A_max: float, per_channel_A: bool, jpeg_quality: int):
    with Image.open(in_path) as im:
        im = im.convert("RGB")
        J = np.asarray(im, dtype=np.float32) / 255.0

    # 随机采样雾强度和大气光
    t = random.uniform(t_min, t_max)
    if per_channel_A:
        A = np.array([random.uniform(A_min, A_max) for _ in range(3)], dtype=np.float32)
    else:
        a = random.uniform(A_min, A_max)
        A = np.array([a, a, a], dtype=np.float32)

    I = haze_image(J, t, A)
    I_img = Image.fromarray((I * 255.0 + 0.5).astype(np.uint8), mode="RGB")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = in_path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        I_img.save(out_path, quality=jpeg_quality, subsampling=1)
    else:
        I_img.save(out_path)

def iter_images(input_dir: Path, recursive: bool):
    if recursive:
        it = input_dir.rglob("*")
    else:
        it = input_dir.glob("*")
    for p in it:
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            yield p

def main():
    parser = argparse.ArgumentParser(
        description="对文件夹内可见光图像按大气散射模型随机加雾，并以相同文件名导出。"
    )
    parser.add_argument("input_dir", type=Path, help="输入图像文件夹路径")
    parser.add_argument("output_dir", type=Path, help="输出文件夹路径（将写入同名文件）")
    parser.add_argument("--t-min", type=float, default=0.2,
                        help="随机传输率下界(0-1，越小雾越浓)，默认0.4")
    parser.add_argument("--t-max", type=float, default=0.95,
                        help="随机传输率上界(0-1)，默认0.95")
    parser.add_argument("--A-min", type=float, default=0.7,
                        help="随机大气光下界(0-1)，默认0.7")
    parser.add_argument("--A-max", type=float, default=1.0,
                        help="随机大气光上界(0-1)，默认1.0")
    parser.add_argument("--per-channel-A", action="store_true",
                        help="为每个通道分别随机A（更真实），默认关闭为各通道同一A")
    parser.add_argument("--recursive", action="store_true",
                        help="递归处理子目录（会在输出端复刻目录结构）")
    parser.add_argument("--seed", type=int, default=None, help="设置随机种子以复现实验")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPG输出质量，默认95")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    input_dir = args.input_dir
    output_dir = args.output_dir

    assert input_dir.exists() and input_dir.is_dir(), "input_dir 必须是存在的文件夹"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 参数合法性
    for name, lo, hi in [("t", args.t_min, args.t_max), ("A", args.A_min, args.A_max)]:
        assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0 and lo < hi, f"{name} 区间需在[0,1]且 lo < hi"

    cnt = 0
    for in_path in iter_images(input_dir, args.recursive):
        # 计算输出路径（保持相同文件名与相对结构）
        rel = in_path.relative_to(input_dir) if args.recursive else in_path.name
        out_path = output_dir / rel
        process_one(in_path, out_path, args.t_min, args.t_max,
                    args.A_min, args.A_max, args.per_channel_A, args.jpeg_quality)
        cnt += 1

    print(f"完成：共处理 {cnt} 张图像。输出路径：{output_dir}")

if __name__ == "__main__":
    main()
