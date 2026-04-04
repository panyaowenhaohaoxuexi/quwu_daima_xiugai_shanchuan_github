import cv2
import numpy as np
from pathlib import Path


# =========================
# 直接在这里指定路径和参数
# =========================
VIS_DIR = r"F:\ACMMM2026\论文里的图优化\ReMix方案图优化\RGB_IR\hazy_RGB"
IR_DIR = r"F:\ACMMM2026\论文里的图优化\ReMix方案图优化\RGB_IR\IR"
OUT_DIR = r"F:\ACMMM2026\论文里的图优化\ReMix方案图优化\RGB_IR\output_dynamic"

# 是否保存可视化结果
SAVE_MASK = True          # 保存浓雾二值区域
SAVE_ALPHA = True         # 保存动态权重图 alpha_map
MASK_SUFFIX = "_dense_mask"
ALPHA_SUFFIX = "_alpha"

# 是否递归读取子文件夹
RECURSIVE = False

# =========================
# 动态融合参数
# =========================
# 1) 用 HSV 估计雾浓度：
# haze_score = w_v * normalized(V) + w_s * normalized(255 - S)
# V 越高、S 越低，越像雾
W_V = 0.98
W_S = 0.05

# 2) 连续分段控制 IR 权重
# haze_score <= CLEAR_TH   -> alpha = 0
# haze_score >= DENSE_TH   -> alpha = 1
# 中间线性/非线性过渡
CLEAR_TH = 0.35
DENSE_TH = 0.90

# 3) 过渡区的衰减形状
# ALPHA_GAMMA > 1：靠近清晰区衰减更快
# ALPHA_GAMMA < 1：更早引入红外
ALPHA_GAMMA = 1.2

# 4) 是否平滑 alpha 图，避免边界生硬
SMOOTH_ALPHA = True
ALPHA_BLUR_KERNEL = 9     # 建议奇数，如 5/7/9/11

# 5) 浓雾区域二值 mask 的阈值（仅用于额外保存 mask，可视化用）
DENSE_MASK_TH = 0.90

# 6) 可选：限制清晰区不被误替换
# 如果一个像素亮度太低，通常不太像“白雾”，可以进一步压低 alpha
ENABLE_DARK_SUPPRESS = False
DARK_V_TH = 40            # V 小于这个值时压制 alpha


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    path = str(path)
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path, img):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == "":
        suffix = ".png"

    success, encoded = cv2.imencode(suffix, img)
    if not success:
        return False
    encoded.tofile(str(path))
    return True


def list_images(folder: Path, recursive=False):
    if recursive:
        files = [p for p in folder.rglob("*") if p.suffix.lower() in VALID_EXTS]
    else:
        files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    return sorted(files)


def build_stem_to_path_dict(files):
    mapping = {}
    for p in files:
        stem = p.stem
        if stem in mapping:
            raise ValueError(f"存在重名文件（同 stem）: {stem}\n{mapping[stem]}\n{p}")
        mapping[stem] = p
    return mapping


def ensure_three_channel_ir(ir_img: np.ndarray):
    if ir_img is None:
        raise ValueError("红外图像为空。")

    if ir_img.ndim == 2:
        ir_gray = ir_img
    elif ir_img.ndim == 3:
        if ir_img.shape[2] == 1:
            ir_gray = ir_img[:, :, 0]
        else:
            ir_gray = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError("不支持的红外图像维度。")

    return cv2.merge([ir_gray, ir_gray, ir_gray])


def compute_haze_score(vis_bgr: np.ndarray):
    """
    haze_score in [0, 1]
    越接近 1，表示越像浓雾区域
    """
    hsv = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    _, s, v = cv2.split(hsv)

    v_norm = v / 255.0
    low_sat_norm = (255.0 - s) / 255.0

    haze_score = W_V * v_norm + W_S * low_sat_norm
    haze_score = np.clip(haze_score, 0.0, 1.0)

    if ENABLE_DARK_SUPPRESS:
        dark_mask = (v < DARK_V_TH).astype(np.float32)
        haze_score = haze_score * (1.0 - dark_mask)

    return haze_score


def compute_alpha_map(haze_score: np.ndarray):
    """
    分段连续权重：
    - 清晰区：alpha = 0
    - 过渡区：alpha 连续上升
    - 浓雾区：alpha = 1
    """
    alpha = np.zeros_like(haze_score, dtype=np.float32)

    # 浓雾区：直接红外替代
    dense_mask = haze_score >= DENSE_TH
    alpha[dense_mask] = 1.0

    # 过渡区：动态连续融合
    trans_mask = (haze_score > CLEAR_TH) & (haze_score < DENSE_TH)
    alpha[trans_mask] = (haze_score[trans_mask] - CLEAR_TH) / (DENSE_TH - CLEAR_TH)
    alpha[trans_mask] = np.power(alpha[trans_mask], ALPHA_GAMMA)

    if SMOOTH_ALPHA and ALPHA_BLUR_KERNEL > 1:
        k = ALPHA_BLUR_KERNEL
        if k % 2 == 0:
            k += 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
        alpha = np.clip(alpha, 0.0, 1.0)

    return alpha


def dynamic_fuse(vis_bgr: np.ndarray, ir_3ch: np.ndarray, alpha_map: np.ndarray):
    """
    out = (1 - alpha) * vis + alpha * ir
    当 alpha=1 时，相当于直接用 IR 替代
    """
    vis_f = vis_bgr.astype(np.float32)
    ir_f = ir_3ch.astype(np.float32)

    alpha_3 = alpha_map[:, :, None]
    out = (1.0 - alpha_3) * vis_f + alpha_3 * ir_f
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def main():
    vis_dir = Path(VIS_DIR)
    ir_dir = Path(IR_DIR)
    out_dir = Path(OUT_DIR)

    if not vis_dir.exists():
        raise FileNotFoundError(f"可见光文件夹不存在: {vis_dir}")
    if not ir_dir.exists():
        raise FileNotFoundError(f"红外文件夹不存在: {ir_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    if SAVE_MASK:
        mask_dir = out_dir / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)

    if SAVE_ALPHA:
        alpha_dir = out_dir / "alphas"
        alpha_dir.mkdir(parents=True, exist_ok=True)

    vis_files = list_images(vis_dir, recursive=RECURSIVE)
    ir_files = list_images(ir_dir, recursive=RECURSIVE)

    if len(vis_files) == 0:
        raise RuntimeError(f"可见光文件夹中没有找到图像: {vis_dir}")
    if len(ir_files) == 0:
        raise RuntimeError(f"红外文件夹中没有找到图像: {ir_dir}")

    vis_map = build_stem_to_path_dict(vis_files)
    ir_map = build_stem_to_path_dict(ir_files)

    common_stems = sorted(set(vis_map.keys()) & set(ir_map.keys()))
    only_vis = sorted(set(vis_map.keys()) - set(ir_map.keys()))
    only_ir = sorted(set(ir_map.keys()) - set(vis_map.keys()))

    print(f"可见光图像数量: {len(vis_files)}")
    print(f"红外图像数量:   {len(ir_files)}")
    print(f"成功匹配数量:   {len(common_stems)}")

    if only_vis:
        print(f"[警告] 有 {len(only_vis)} 张可见光图像没有匹配到红外图。")
    if only_ir:
        print(f"[警告] 有 {len(only_ir)} 张红外图像没有匹配到可见光图。")

    if len(common_stems) == 0:
        raise RuntimeError("没有找到同名配对图像。请检查两个文件夹中的文件名是否一致。")

    for idx, stem in enumerate(common_stems, start=1):
        vis_path = vis_map[stem]
        ir_path = ir_map[stem]

        vis_bgr = imread_unicode(vis_path, cv2.IMREAD_COLOR)
        ir_img = imread_unicode(ir_path, cv2.IMREAD_UNCHANGED)

        if vis_bgr is None:
            print(f"[跳过] 读取可见光图像失败: {vis_path}")
            continue
        if ir_img is None:
            print(f"[跳过] 读取红外图像失败: {ir_path}")
            continue

        ir_3ch = ensure_three_channel_ir(ir_img)

        if vis_bgr.shape[:2] != ir_3ch.shape[:2]:
            ir_3ch = cv2.resize(
                ir_3ch,
                (vis_bgr.shape[1], vis_bgr.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )

        haze_score = compute_haze_score(vis_bgr)
        alpha_map = compute_alpha_map(haze_score)
        out_img = dynamic_fuse(vis_bgr, ir_3ch, alpha_map)

        out_path = out_dir / vis_path.name
        ok = imwrite_unicode(out_path, out_img)
        if not ok:
            print(f"[跳过] 保存结果失败: {out_path}")
            continue

        if SAVE_MASK:
            dense_mask = (alpha_map >= DENSE_MASK_TH).astype(np.uint8) * 255
            mask_name = f"{vis_path.stem}{MASK_SUFFIX}.png"
            imwrite_unicode(mask_dir / mask_name, dense_mask)

        if SAVE_ALPHA:
            alpha_vis = (alpha_map * 255.0).clip(0, 255).astype(np.uint8)
            alpha_name = f"{vis_path.stem}{ALPHA_SUFFIX}.png"
            imwrite_unicode(alpha_dir / alpha_name, alpha_vis)

        print(f"[{idx}/{len(common_stems)}] 已保存: {out_path}")

    print("处理完成。")


if __name__ == "__main__":
    main()