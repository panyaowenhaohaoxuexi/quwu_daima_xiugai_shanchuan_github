import cv2
import numpy as np
from pathlib import Path


# =========================
# 直接在这里指定路径
# =========================
VIS_DIR = r"F:\ACMMM2026\论文里的图优化\ReMix方案图优化\RGB_IR\hazy_RGB"
IR_DIR = r"F:\ACMMM2026\论文里的图优化\ReMix方案图优化\RGB_IR\IR"
OUT_DIR = r"F:\ACMMM2026\论文里的图优化\ReMix方案图优化\RGB_IR\grid_results"

RECURSIVE = False

# =========================
# 是否保存辅助结果
# =========================
SAVE_MASK = True
SAVE_ALPHA = True
MASK_SUFFIX = "_dense_mask"
ALPHA_SUFFIX = "_alpha"

# =========================
# 自动生成 500 组参数
# =========================
NUM_PARAM_SETS = 500
RANDOM_SEED = 2026

# 雾浓度估计参数范围
# haze_score = w_v * V_norm + w_s * (1 - S_norm)
WV_MIN, WV_MAX = 0.30, 0.80          # W_V
CLEAR_TH_MIN, CLEAR_TH_MAX = 0.20, 0.65
DENSE_MARGIN_MIN, DENSE_MARGIN_MAX = 0.10, 0.35   # dense_th = clear_th + margin
GAMMA_MIN, GAMMA_MAX = 0.50, 2.50
BLUR_KERNEL_CANDIDATES = [0, 3, 5, 7, 9, 11, 13, 15]

# 保留小数位
ROUND_DECIMALS = 3

# 仅用于保存 dense mask 的阈值
DENSE_MASK_TH = 0.90

# 暗区抑制（可选）
ENABLE_DARK_SUPPRESS = False
DARK_V_TH = 40

# =========================
# 按图像聚合对比结果
# =========================
ENABLE_COMPARE_BY_IMAGE = True
COMPARE_DIR_NAME = "compare_by_image"

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


def compute_haze_score(vis_bgr: np.ndarray, w_v: float, w_s: float, enable_dark_suppress: bool, dark_v_th: int):
    """
    haze_score in [0, 1]
    越接近 1，表示越像浓雾区域
    """
    hsv = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    _, s, v = cv2.split(hsv)

    v_norm = v / 255.0
    low_sat_norm = (255.0 - s) / 255.0

    haze_score = w_v * v_norm + w_s * low_sat_norm
    haze_score = np.clip(haze_score, 0.0, 1.0)

    if enable_dark_suppress:
        dark_mask = (v < dark_v_th).astype(np.float32)
        haze_score = haze_score * (1.0 - dark_mask)

    return haze_score


def compute_alpha_map(haze_score: np.ndarray, clear_th: float, dense_th: float, alpha_gamma: float, blur_kernel: int):
    """
    分段连续权重：
    - 清晰区：alpha = 0
    - 过渡区：alpha 连续上升
    - 浓雾区：alpha = 1
    """
    if dense_th <= clear_th:
        raise ValueError(f"dense_th 必须大于 clear_th, 当前: clear_th={clear_th}, dense_th={dense_th}")

    alpha = np.zeros_like(haze_score, dtype=np.float32)

    dense_mask = haze_score >= dense_th
    alpha[dense_mask] = 1.0

    trans_mask = (haze_score > clear_th) & (haze_score < dense_th)
    alpha[trans_mask] = (haze_score[trans_mask] - clear_th) / (dense_th - clear_th)
    alpha[trans_mask] = np.power(alpha[trans_mask], alpha_gamma)

    if blur_kernel and blur_kernel > 1:
        k = int(blur_kernel)
        if k % 2 == 0:
            k += 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)
        alpha = np.clip(alpha, 0.0, 1.0)

    return alpha


def dynamic_fuse(vis_bgr: np.ndarray, ir_3ch: np.ndarray, alpha_map: np.ndarray):
    """
    out = (1 - alpha) * vis + alpha * ir
    alpha=1 时，直接用 IR 替代
    """
    vis_f = vis_bgr.astype(np.float32)
    ir_f = ir_3ch.astype(np.float32)

    alpha_3 = alpha_map[:, :, None]
    out = (1.0 - alpha_3) * vis_f + alpha_3 * ir_f
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def make_run_name(run_idx, w_v, w_s, clear_th, dense_th, gamma, blur_k):
    return (
        f"run_{run_idx:03d}"
        f"_wv{w_v:.3f}"
        f"_ws{w_s:.3f}"
        f"_ct{clear_th:.3f}"
        f"_dt{dense_th:.3f}"
        f"_g{gamma:.3f}"
        f"_bk{int(blur_k)}"
    )


def save_params_txt(folder: Path, params: dict):
    lines = []
    for k, v in params.items():
        lines.append(f"{k} = {v}")
    (folder / "params.txt").write_text("\n".join(lines), encoding="utf-8")


def generate_random_param_sets(num_sets=500, seed=2026):
    rng = np.random.default_rng(seed)
    param_set = set()

    max_tries = num_sets * 100
    tries = 0

    while len(param_set) < num_sets and tries < max_tries:
        tries += 1

        # 1) 采样 w_v, w_s，保证两者和为 1
        w_v = float(rng.uniform(WV_MIN, WV_MAX))
        w_s = 1.0 - w_v

        # 2) 采样 clear_th
        clear_th = float(rng.uniform(CLEAR_TH_MIN, CLEAR_TH_MAX))

        # 3) 采样 dense_th，保证 dense_th > clear_th
        margin = float(rng.uniform(DENSE_MARGIN_MIN, DENSE_MARGIN_MAX))
        dense_th = clear_th + margin
        dense_th = min(dense_th, 0.95)

        if dense_th <= clear_th:
            continue

        # 4) 采样 gamma
        gamma = float(rng.uniform(GAMMA_MIN, GAMMA_MAX))

        # 5) 采样 blur kernel
        blur_k = int(rng.choice(BLUR_KERNEL_CANDIDATES))

        # 6) 四舍五入
        w_v = round(w_v, ROUND_DECIMALS)
        w_s = round(1.0 - w_v, ROUND_DECIMALS)
        clear_th = round(clear_th, ROUND_DECIMALS)
        dense_th = round(dense_th, ROUND_DECIMALS)
        gamma = round(gamma, ROUND_DECIMALS)

        # 最终合法性检查
        if not (0.0 <= w_v <= 1.0 and 0.0 <= w_s <= 1.0):
            continue
        if abs((w_v + w_s) - 1.0) > 1e-3:
            continue
        if dense_th <= clear_th:
            continue

        param_set.add((w_v, w_s, clear_th, dense_th, gamma, blur_k))

    if len(param_set) < num_sets:
        raise RuntimeError(
            f"只生成了 {len(param_set)} 组有效参数，未达到 {num_sets} 组。"
            f"请增大参数范围或提高 max_tries。"
        )

    return sorted(list(param_set))


def append_compare_index(index_path: Path, line: str):
    with open(index_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    vis_dir = Path(VIS_DIR)
    ir_dir = Path(IR_DIR)
    out_dir = Path(OUT_DIR)

    if not vis_dir.exists():
        raise FileNotFoundError(f"可见光文件夹不存在: {vis_dir}")
    if not ir_dir.exists():
        raise FileNotFoundError(f"红外文件夹不存在: {ir_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

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

    # compare_by_image 初始化
    compare_root = None
    if ENABLE_COMPARE_BY_IMAGE:
        compare_root = out_dir / COMPARE_DIR_NAME
        compare_root.mkdir(parents=True, exist_ok=True)

        for stem in common_stems:
            image_compare_dir = compare_root / stem
            image_compare_dir.mkdir(parents=True, exist_ok=True)
            index_path = image_compare_dir / "index.txt"
            index_path.write_text(
                "file_name | run_name | W_V | W_S | CLEAR_TH | DENSE_TH | ALPHA_GAMMA | ALPHA_BLUR_KERNEL\n",
                encoding="utf-8"
            )

    valid_params = generate_random_param_sets(
        num_sets=NUM_PARAM_SETS,
        seed=RANDOM_SEED
    )

    print(f"有效参数组合数量: {len(valid_params)}")

    for run_idx, (w_v, w_s, clear_th, dense_th, gamma, blur_k) in enumerate(valid_params, start=1):
        run_name = make_run_name(run_idx, w_v, w_s, clear_th, dense_th, gamma, blur_k)
        run_dir = out_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)

        if SAVE_MASK:
            mask_dir = run_dir / "masks"
            mask_dir.mkdir(parents=True, exist_ok=True)

        if SAVE_ALPHA:
            alpha_dir = run_dir / "alphas"
            alpha_dir.mkdir(parents=True, exist_ok=True)

        params = {
            "VIS_DIR": VIS_DIR,
            "IR_DIR": IR_DIR,
            "W_V": w_v,
            "W_S": w_s,
            "CLEAR_TH": clear_th,
            "DENSE_TH": dense_th,
            "ALPHA_GAMMA": gamma,
            "ALPHA_BLUR_KERNEL": blur_k,
            "DENSE_MASK_TH": DENSE_MASK_TH,
            "ENABLE_DARK_SUPPRESS": ENABLE_DARK_SUPPRESS,
            "DARK_V_TH": DARK_V_TH,
            "SAVE_MASK": SAVE_MASK,
            "SAVE_ALPHA": SAVE_ALPHA,
            "NUM_PARAM_SETS": NUM_PARAM_SETS,
            "RANDOM_SEED": RANDOM_SEED,
        }
        save_params_txt(run_dir, params)

        print(f"\n===== 开始处理参数组 {run_idx}/{len(valid_params)}: {run_name} =====")

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

            haze_score = compute_haze_score(
                vis_bgr=vis_bgr,
                w_v=w_v,
                w_s=w_s,
                enable_dark_suppress=ENABLE_DARK_SUPPRESS,
                dark_v_th=DARK_V_TH,
            )

            alpha_map = compute_alpha_map(
                haze_score=haze_score,
                clear_th=clear_th,
                dense_th=dense_th,
                alpha_gamma=gamma,
                blur_kernel=blur_k,
            )

            out_img = dynamic_fuse(vis_bgr, ir_3ch, alpha_map)

            # 1) 保存到参数组文件夹
            out_path = run_dir / vis_path.name
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

            # 2) 额外保存到 compare_by_image/图像名/ 下
            if ENABLE_COMPARE_BY_IMAGE:
                image_compare_dir = compare_root / stem
                compare_name = f"{run_name}.png"
                compare_path = image_compare_dir / compare_name
                imwrite_unicode(compare_path, out_img)

                index_path = image_compare_dir / "index.txt"
                append_compare_index(
                    index_path,
                    f"{compare_name} | {run_name} | {w_v} | {w_s} | {clear_th} | {dense_th} | {gamma} | {blur_k}"
                )

            print(f"[参数组 {run_idx}/{len(valid_params)}] [图像 {idx}/{len(common_stems)}] 已保存: {out_path}")

    print("\n全部处理完成。")
    if ENABLE_COMPARE_BY_IMAGE:
        print(f"按图像汇总的对比结果已保存到: {out_dir / COMPARE_DIR_NAME}")


if __name__ == "__main__":
    main()