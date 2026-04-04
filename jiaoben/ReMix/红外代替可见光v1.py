import cv2
import numpy as np
from pathlib import Path


# =========================
# 直接在这里指定路径和参数
# =========================
VIS_DIR = r"F:\ACMMM2026\论文里的图优化\ReMix方案图优化\RGB_IR\hazy_RGB"
IR_DIR = r"F:\ACMMM2026\论文里的图优化\ReMix方案图优化\RGB_IR\IR"
OUT_DIR = r"F:\ACMMM2026\论文里的图优化\ReMix方案图优化\RGB_IR\output"

# 浓雾区域判定阈值
VALUE_THRESH = 170
SAT_THRESH = 70    # 设为 -1 表示不用饱和度约束

# 替换模式
MODE = "replace"   # "replace" 或 "blend"
# MODE = "blend"   # "replace" 或 "blend"
ALPHA = 0.85        # 仅在 blend 模式下有效

# 是否保存 mask
SAVE_MASK = True
MASK_SUFFIX = "_mask"

# mask 后处理
KERNEL_SIZE = 5

# 是否递归读取子文件夹
RECURSIVE = False


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    path = str(path)
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    img = cv2.imdecode(data, flags)
    return img


def imwrite_unicode(path, img):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == "":
        suffix = ".png"

    ext = suffix
    success, encoded_img = cv2.imencode(ext, img)
    if not success:
        return False

    encoded_img.tofile(str(path))
    return True


def list_images(folder: Path, recursive: bool = False):
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

    ir_3ch = cv2.merge([ir_gray, ir_gray, ir_gray])
    return ir_3ch


def compute_dense_haze_mask(vis_bgr: np.ndarray, value_thresh: int, sat_thresh: int, kernel_size: int):
    hsv = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    mask_v = (v >= value_thresh)
    if sat_thresh >= 0:
        mask_s = (s <= sat_thresh)
        mask = (mask_v & mask_s).astype(np.uint8) * 255
    else:
        mask = mask_v.astype(np.uint8) * 255

    if kernel_size > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def replace_or_blend(vis_bgr: np.ndarray, ir_3ch: np.ndarray, mask: np.ndarray, mode: str, alpha: float):
    mask_bool = mask > 0
    out = vis_bgr.copy()

    if mode == "replace":
        out[mask_bool] = ir_3ch[mask_bool]
    elif mode == "blend":
        alpha = float(np.clip(alpha, 0.0, 1.0))
        fused = cv2.addWeighted(ir_3ch, alpha, vis_bgr, 1.0 - alpha, 0.0)
        out[mask_bool] = fused[mask_bool]
    else:
        raise ValueError("MODE 只能是 'replace' 或 'blend'。")

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

        mask = compute_dense_haze_mask(
            vis_bgr=vis_bgr,
            value_thresh=VALUE_THRESH,
            sat_thresh=SAT_THRESH,
            kernel_size=KERNEL_SIZE,
        )

        out_img = replace_or_blend(
            vis_bgr=vis_bgr,
            ir_3ch=ir_3ch,
            mask=mask,
            mode=MODE,
            alpha=ALPHA,
        )

        out_path = out_dir / vis_path.name
        ok = imwrite_unicode(out_path, out_img)
        if not ok:
            print(f"[跳过] 保存结果失败: {out_path}")
            continue

        if SAVE_MASK:
            mask_name = f"{vis_path.stem}{MASK_SUFFIX}.png"
            mask_path = mask_dir / mask_name
            imwrite_unicode(mask_path, mask)

        print(f"[{idx}/{len(common_stems)}] 已保存: {out_path}")

    print("处理完成。")


if __name__ == "__main__":
    main()