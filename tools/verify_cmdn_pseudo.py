"""
Offline verification for CMDN fixed pseudo-label P_pseudo.

Edit HAZY_DIR, IR_DIR, OUT_DIR, SIZE, and MAX_IMAGES below, then run:

python tools/verify_cmdn_pseudo.py

HAZY_DIR and IR_DIR may be either folders or matching single image files.
OUT_DIR may be a folder, or a .png path for a single debug visualization.

This script does not train, create an optimizer, compute losses, call
backward, or load the Teacher/VIFNet model. It only runs CMDN with
return_debug=True and saves pseudo-label diagnostics. SAM is used only in this
offline script to produce sky_mask prompts for CMDN sky suppression.
"""

import csv
import importlib.util
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# User-editable configuration
# ---------------------------------------------------------------------------

HAZY_DIR = r"F:/Dehaze_Paper/2_Dataset/2_additional_benchmark/Target/hazy/dense_hazy_01.jpg"
IR_DIR = r"F:/Dehaze_Paper/2_Dataset/2_additional_benchmark/Target/ir/dense_hazy_01.jpg"
OUT_DIR = r"D:/liu_lan_qi_xia_zai/quwu_daima_xiugai_shanchuan_github/cmdn_verify_results/dense_hazy_01.jpg"

SIZE = 512
MAX_IMAGES = 20
DEVICE = "cuda"

DINO_SOURCE_DIR = r"./DINOv2/facebookresearch_dinov2_main"
DINO_WEIGHT_PATH = r"./dinov2_model/dinov2_vitb14_pretrain.pth"

USE_SAM_SKY_MASK = True

SAM_CHECKPOINT = r"./sam_model/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"

SKY_MASK_SAVE_DIRNAME = "sky_mask"
SKY_OVERLAY_SAVE_DIRNAME = "sky_overlay"

SKY_POS_POINT_NUM = 5
SKY_NEG_POINT_NUM = 5
SKY_PRIOR_TOP_RATIO = 0.55
SKY_MASK_THRESHOLD = 0.5

SKY_MASK_TOO_LARGE = 0.65
SKY_MASK_TOO_SMALL = 0.01

EXTENSIONS = [".jpg", ".png", ".jpeg"]


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_cmdn_class():
    cmdn_path = PROJECT_ROOT / "model" / "cmdn.py"
    spec = importlib.util.spec_from_file_location("cmdn_direct", cmdn_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load CMDN module from: {cmdn_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CMDN


CMDN = load_cmdn_class()


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)

REQUIRED_DEBUG_KEYS = [
    "g_fog",
    "haze_app",
    "attn_deg",
    "struct_deg",
    "ir_adv",
    "P_pseudo_raw",
    "sky_mask",
    "non_sky_mask",
    "P_pseudo",
    "P_fail",
    "tau",
    "G_dec",
    "P_support",
    "G_soft",
    "M_hard",
]

CSV_FIELDS = [
    "filename",
    "p_min",
    "p_mean",
    "p_max",
    "sky_mask_mean",
    "p_raw_min",
    "p_raw_mean",
    "p_raw_max",
    "p_final_min",
    "p_final_mean",
    "p_final_max",
    "suppression_ratio",
    "m_hard_mean",
    "g_soft_mean",
    "p_fail_mean",
    "tau_mean",
    "max_abs_alpha_diff",
    "requires_grad",
]


def resolve_device():
    requested = DEVICE.lower()
    if requested == "cuda" and not torch.cuda.is_available():
        print("DEVICE='cuda' requested, but CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def load_sam_predictor(device):
    if not USE_SAM_SKY_MASK:
        return None

    checkpoint = Path(SAM_CHECKPOINT)
    if not checkpoint.exists():
        raise RuntimeError(f"SAM checkpoint does not exist: {checkpoint}")

    try:
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError as exc:
        raise RuntimeError(
            "segment_anything is required when USE_SAM_SKY_MASK=True") from exc

    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(checkpoint))
    sam.to(device)
    sam.eval()
    return SamPredictor(sam)


def is_supported_image(path):
    return path.is_file() and path.suffix.lower() in EXTENSIONS


def ensure_input_paths(hazy_path, ir_path):
    if not hazy_path.exists():
        raise RuntimeError(f"HAZY_DIR does not exist: {hazy_path}")
    if not ir_path.exists():
        raise RuntimeError(f"IR_DIR does not exist: {ir_path}")
    if hazy_path.is_file() and not is_supported_image(hazy_path):
        raise RuntimeError(f"HAZY_DIR file has unsupported extension: {hazy_path}")
    if ir_path.is_file() and not is_supported_image(ir_path):
        raise RuntimeError(f"IR_DIR file has unsupported extension: {ir_path}")
    if not hazy_path.is_file() and not hazy_path.is_dir():
        raise RuntimeError(f"HAZY_DIR is neither a file nor a directory: {hazy_path}")
    if not ir_path.is_file() and not ir_path.is_dir():
        raise RuntimeError(f"IR_DIR is neither a file nor a directory: {ir_path}")


def collect_hazy_images(hazy_dir):
    if hazy_dir.is_file():
        return [hazy_dir]

    images = sorted([p for p in hazy_dir.iterdir() if is_supported_image(p)],
                    key=lambda p: p.name.lower())
    if not images:
        raise RuntimeError(
            f"HAZY_DIR contains no supported images ({EXTENSIONS}): {hazy_dir}")
    return images


def build_ir_index(ir_dir):
    if ir_dir.is_file():
        return {ir_dir.name.lower(): ir_dir}, {ir_dir.stem.lower(): ir_dir}

    ir_files = sorted([p for p in ir_dir.iterdir() if is_supported_image(p)],
                      key=lambda p: p.name.lower())
    by_name = {p.name.lower(): p for p in ir_files}
    by_stem = {}
    for p in ir_files:
        by_stem.setdefault(p.stem.lower(), p)
    return by_name, by_stem


def find_ir_image(hazy_path, ir_by_name, ir_by_stem):
    exact = ir_by_name.get(hazy_path.name.lower())
    if exact is not None:
        return exact
    return ir_by_stem.get(hazy_path.stem.lower())


def load_rgb(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((SIZE, SIZE), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return (tensor - CLIP_MEAN) / CLIP_STD


def denorm_clip_for_display(x):
    mean = CLIP_MEAN.to(device=x.device, dtype=x.dtype).unsqueeze(0)
    std = CLIP_STD.to(device=x.device, dtype=x.dtype).unsqueeze(0)
    img = (x * std + mean).clamp(0.0, 1.0)[0]
    return img.detach().cpu().permute(1, 2, 0).numpy()


def to_numpy_rgb01(vis_01):
    if isinstance(vis_01, torch.Tensor):
        arr = vis_01.detach().cpu().float().numpy()
        if arr.ndim == 3 and arr.shape[0] == 3:
            arr = np.transpose(arr, (1, 2, 0))
    else:
        arr = np.asarray(vis_01, dtype=np.float32)
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def compute_sky_prior(vis_01):
    vis = to_numpy_rgb01(vis_01)
    r, g, b = vis[..., 0], vis[..., 1], vis[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    saturation = vis.max(axis=2) - vis.min(axis=2)

    gray_t = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0)
    mean = F.avg_pool2d(gray_t, kernel_size=9, stride=1, padding=4)
    mean_sq = F.avg_pool2d(gray_t * gray_t, kernel_size=9, stride=1, padding=4)
    local_std = torch.sqrt((mean_sq - mean * mean).clamp_min(0.0) + 1e-8)
    local_std = local_std.squeeze().numpy()

    h, _ = gray.shape
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    upper_prior = np.clip(1.0 - y / max(SKY_PRIOR_TOP_RATIO, 1e-6), 0.0, 1.0)

    bright_gate = sigmoid_np((gray - 0.45) / 0.12)
    low_sat_gate = sigmoid_np((0.40 - saturation) / 0.10)
    low_texture_gate = sigmoid_np((0.08 - local_std) / 0.04)
    sky_prior = bright_gate * low_sat_gate * low_texture_gate * upper_prior
    return np.clip(sky_prior.astype(np.float32), 0.0, 1.0)


def select_spaced_points(candidates, count, min_dist):
    selected = []
    min_dist_sq = float(min_dist * min_dist)
    for y, x in candidates:
        if all((float(y - sy) ** 2 + float(x - sx) ** 2) >= min_dist_sq
               for sy, sx in selected):
            selected.append((int(y), int(x)))
            if len(selected) >= count:
                break
    return selected


def select_sam_points(sky_prior):
    h, w = sky_prior.shape
    top_h = max(1, int(round(h * SKY_PRIOR_TOP_RATIO)))
    min_dist = max(8, min(h, w) // 8)

    top_scores = sky_prior[:top_h, :]
    pos_flat = np.argsort(top_scores.reshape(-1))[::-1]
    pos_candidates = [(idx // w, idx % w) for idx in pos_flat]
    pos_points = select_spaced_points(pos_candidates, SKY_POS_POINT_NUM, min_dist)

    bottom = sky_prior[h // 2:, :]
    neg_flat = np.argsort(bottom.reshape(-1))
    neg_candidates = [(h // 2 + idx // w, idx % w) for idx in neg_flat]
    fixed_neg = [
        (int(h * 0.92), int(w * 0.12)),
        (int(h * 0.92), int(w * 0.50)),
        (int(h * 0.92), int(w * 0.88)),
        (int(h * 0.68), int(w * 0.25)),
        (int(h * 0.68), int(w * 0.75)),
    ]
    neg_points = select_spaced_points(fixed_neg + neg_candidates,
                                      SKY_NEG_POINT_NUM, min_dist)

    points = pos_points + neg_points
    if not points:
        raise RuntimeError("No SAM prompt points were selected.")

    point_coords = np.array([[x, y] for y, x in points], dtype=np.float32)
    point_labels = np.array(
        [1] * len(pos_points) + [0] * len(neg_points), dtype=np.int32)
    return point_coords, point_labels


def keep_top_connected_components(mask):
    mask_bool = mask.astype(bool)
    h, w = mask_bool.shape
    visited = np.zeros_like(mask_bool, dtype=bool)
    keep = np.zeros_like(mask_bool, dtype=bool)
    queue = deque()

    for x in range(w):
        if mask_bool[0, x]:
            queue.append((0, x))
            visited[0, x] = True

    while queue:
        y, x = queue.popleft()
        keep[y, x] = True
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and mask_bool[ny, nx] and not visited[ny, nx]:
                visited[ny, nx] = True
                queue.append((ny, nx))

    if keep.any():
        return keep.astype(np.float32)
    return mask.astype(np.float32)


def generate_sam_sky_mask(predictor, vis_uint8, sky_prior):
    point_coords, point_labels = select_sam_points(sky_prior)
    predictor.set_image(vis_uint8)
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    if masks is None or len(masks) == 0:
        raise RuntimeError("SAM returned no masks.")

    h, _ = sky_prior.shape
    best_idx = 0
    best_score = -float("inf")
    for idx, mask in enumerate(masks):
        mask_bool = mask.astype(bool)
        if mask_bool.any():
            prior_inside = float(sky_prior[mask_bool].mean())
        else:
            prior_inside = 0.0
        top_border_touch_ratio = float(mask_bool[0, :].mean())
        bottom_coverage = float(mask_bool[int(h * 0.75):, :].mean())
        sam_pred_score = float(scores[idx]) if scores is not None else 0.0
        score = (sam_pred_score + 0.5 * prior_inside +
                 0.3 * top_border_touch_ratio - 0.5 * bottom_coverage)
        if score > best_score:
            best_score = score
            best_idx = idx

    sky_mask = masks[best_idx].astype(np.float32)
    sky_mask = (sky_mask >= SKY_MASK_THRESHOLD).astype(np.float32)
    sky_mask = keep_top_connected_components(sky_mask)
    return sky_mask.astype(np.float32)


def save_sky_mask(path, sky_mask):
    img = Image.fromarray((np.clip(sky_mask, 0.0, 1.0) * 255).astype(np.uint8))
    img.save(path)


def save_sky_overlay(path, vis_01, sky_mask):
    vis = to_numpy_rgb01(vis_01)
    mask = np.clip(sky_mask[..., None], 0.0, 1.0)
    color = np.array([0.1, 0.65, 1.0], dtype=np.float32)
    overlay = vis * (1.0 - 0.45 * mask) + color * (0.45 * mask)
    Image.fromarray((np.clip(overlay, 0.0, 1.0) * 255).astype(np.uint8)).save(path)


def map_to_numpy(tensor):
    arr = tensor.detach().cpu().float().squeeze().numpy()
    if arr.ndim == 0:
        return float(arr)
    return arr


def validate_debug_keys(debug, filename):
    missing = [key for key in REQUIRED_DEBUG_KEYS if key not in debug]
    if missing:
        raise KeyError(
            f"CMDN return_debug is missing keys for {filename}: {', '.join(missing)}")


def scalar_mean(tensor):
    return float(tensor.detach().cpu().float().mean().item())


def tensor_min_mean_max(tensor):
    x = tensor.detach().cpu().float()
    return float(x.min().item()), float(x.mean().item()), float(x.max().item())


def plot_heat(ax, data, title, cmap="hot"):
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def save_debug_figure(out_path, filename, x_vis, x_ir, debug, sky_prior):
    vis_01 = denorm_clip_for_display(x_vis)
    ir_01 = denorm_clip_for_display(x_ir)

    maps = {
        "sky_prior": sky_prior,
        "sky_mask SAM": map_to_numpy(debug["sky_mask"]),
        "non_sky_mask": map_to_numpy(debug["non_sky_mask"]),
        "g_fog diagnostic": map_to_numpy(debug["g_fog"]),
        "haze_app": map_to_numpy(debug["haze_app"]),
        "attn_deg": map_to_numpy(debug["attn_deg"]),
        "struct_deg": map_to_numpy(debug["struct_deg"]),
        "ir_adv": map_to_numpy(debug["ir_adv"]),
        "P_pseudo_raw": map_to_numpy(debug["P_pseudo_raw"]),
        "P_pseudo SAM": map_to_numpy(debug["P_pseudo"]),
        "P_fail": map_to_numpy(debug["P_fail"]),
        "G_dec": map_to_numpy(debug["G_dec"]),
        "P_support": map_to_numpy(debug["P_support"]),
        "G_soft": map_to_numpy(debug["G_soft"]),
        "M_hard": map_to_numpy(debug["M_hard"]),
    }

    p_fail_map = maps["P_fail"]
    tau_value = scalar_mean(debug["tau"])
    tau_map = np.full_like(p_fail_map, tau_value, dtype=np.float32)

    fig, axes = plt.subplots(1, 18, figsize=(64, 4), squeeze=False)
    axes = axes[0]

    axes[0].imshow(vis_01)
    axes[0].set_title("Hazy Visible", fontsize=8)
    axes[0].axis("off")

    axes[1].imshow(ir_01)
    axes[1].set_title("Infrared", fontsize=8)
    axes[1].axis("off")

    plot_heat(axes[2], maps["sky_prior"], "sky_prior")
    plot_heat(axes[3], maps["sky_mask SAM"], "sky_mask SAM")
    plot_heat(axes[4], maps["non_sky_mask"], "non_sky_mask")
    plot_heat(axes[5], maps["g_fog diagnostic"], "g_fog diagnostic")
    plot_heat(axes[6], maps["haze_app"], "haze_app")
    plot_heat(axes[7], maps["attn_deg"], "attn_deg")
    plot_heat(axes[8], maps["struct_deg"], "struct_deg")
    plot_heat(axes[9], maps["ir_adv"], "ir_adv")
    plot_heat(axes[10], maps["P_pseudo_raw"], "P_pseudo_raw")
    plot_heat(axes[11], maps["P_pseudo SAM"], "P_pseudo SAM")
    plot_heat(axes[12], maps["P_fail"], "P_fail")
    plot_heat(axes[13], tau_map, "tau", cmap="viridis")
    plot_heat(axes[14], maps["G_dec"], "G_dec")
    plot_heat(axes[15], maps["P_support"], "P_support")
    plot_heat(axes[16], maps["G_soft"], "G_soft")
    plot_heat(axes[17], maps["M_hard"], "M_hard", cmap="gray")

    fig.suptitle(filename, fontsize=11, y=1.05)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def stats_for_image(filename, debug0, debug1):
    p_raw = debug0["P_pseudo_raw"]
    p_final = debug0["P_pseudo"]
    raw_min, raw_mean, raw_max = tensor_min_mean_max(p_raw)
    final_min, final_mean, final_max = tensor_min_mean_max(p_final)
    suppression_ratio = 1.0 - (final_mean / (raw_mean + 1e-8))
    max_abs_alpha_diff = (debug0["P_pseudo"] - debug1["P_pseudo"]).abs().max().item()

    if max_abs_alpha_diff > 1e-6:
        print("WARNING: P_pseudo changes with disc_alpha; fixed pseudo-label assumption is violated.")

    sky_mask_mean = scalar_mean(debug0["sky_mask"])
    if sky_mask_mean > SKY_MASK_TOO_LARGE:
        print("WARNING: sky_mask may be too large; non-sky fog regions may be suppressed.")
    if sky_mask_mean < SKY_MASK_TOO_SMALL:
        print("WARNING: sky_mask may be too small; sky may not be suppressed.")

    row = {
        "filename": filename,
        "p_min": final_min,
        "p_mean": final_mean,
        "p_max": final_max,
        "sky_mask_mean": sky_mask_mean,
        "p_raw_min": raw_min,
        "p_raw_mean": raw_mean,
        "p_raw_max": raw_max,
        "p_final_min": final_min,
        "p_final_mean": final_mean,
        "p_final_max": final_max,
        "suppression_ratio": float(suppression_ratio),
        "m_hard_mean": scalar_mean(debug0["M_hard"]),
        "g_soft_mean": scalar_mean(debug0["G_soft"]),
        "p_fail_mean": scalar_mean(debug0["P_fail"]),
        "tau_mean": scalar_mean(debug0["tau"]),
        "max_abs_alpha_diff": float(max_abs_alpha_diff),
        "requires_grad": bool(p_final.requires_grad),
    }
    return row


def print_stats(row):
    print("-" * 80)
    print(f"filename: {row['filename']}")
    print(f"sky_mask mean: {row['sky_mask_mean']:.6f}")
    print(
        "P_pseudo_raw min/mean/max: "
        f"{row['p_raw_min']:.6f} / {row['p_raw_mean']:.6f} / {row['p_raw_max']:.6f}")
    print(
        "P_pseudo final min/mean/max: "
        f"{row['p_final_min']:.6f} / {row['p_final_mean']:.6f} / {row['p_final_max']:.6f}")
    print(f"M_hard mean: {row['m_hard_mean']:.6f}")
    print(f"G_soft mean: {row['g_soft_mean']:.6f}")
    print(f"P_fail mean: {row['p_fail_mean']:.6f}")
    print(f"tau mean: {row['tau_mean']:.6f}")
    print(f"disc_alpha stability max_abs: {row['max_abs_alpha_diff']:.10f}")
    print(f"P_pseudo.requires_grad: {row['requires_grad']}")
    print(f"suppression_ratio: {row['suppression_ratio']:.6f}")


def save_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def save_summary(path, rows):
    pseudo_means = [row["p_mean"] for row in rows]
    pseudo_maxes = [row["p_max"] for row in rows]
    m_hard_means = [row["m_hard_mean"] for row in rows]
    alpha_diffs = [row["max_abs_alpha_diff"] for row in rows]
    sky_mask_means = [row["sky_mask_mean"] for row in rows]
    raw_means = [row["p_raw_mean"] for row in rows]
    raw_maxes = [row["p_raw_max"] for row in rows]
    final_means = [row["p_final_mean"] for row in rows]
    final_maxes = [row["p_final_max"] for row in rows]
    suppression_ratios = [row["suppression_ratio"] for row in rows]

    lines = [
        f"num_images={len(rows)}",
        f"mean_pseudo_mean={float(np.mean(pseudo_means)):.10f}",
        f"mean_pseudo_max={float(np.mean(pseudo_maxes)):.10f}",
        f"mean_m_hard_mean={float(np.mean(m_hard_means)):.10f}",
        f"mean_alpha_diff={float(np.mean(alpha_diffs)):.10f}",
        f"mean_sky_mask_mean={float(np.mean(sky_mask_means)):.10f}",
        f"mean_pseudo_raw_mean={float(np.mean(raw_means)):.10f}",
        f"mean_pseudo_raw_max={float(np.mean(raw_maxes)):.10f}",
        f"mean_pseudo_final_mean={float(np.mean(final_means)):.10f}",
        f"mean_pseudo_final_max={float(np.mean(final_maxes)):.10f}",
        f"mean_suppression_ratio={float(np.mean(suppression_ratios)):.10f}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    hazy_dir = Path(HAZY_DIR)
    ir_dir = Path(IR_DIR)
    out_config = Path(OUT_DIR)
    single_png_out = out_config if out_config.suffix.lower() == ".png" else None
    out_dir = out_config.parent if single_png_out is not None else out_config
    vis_dir = out_dir / "vis"
    sky_mask_dir = out_dir / SKY_MASK_SAVE_DIRNAME
    sky_overlay_dir = out_dir / SKY_OVERLAY_SAVE_DIRNAME

    ensure_input_paths(hazy_dir, ir_dir)
    hazy_images = collect_hazy_images(hazy_dir)
    ir_by_name, ir_by_stem = build_ir_index(ir_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    if USE_SAM_SKY_MASK:
        sky_mask_dir.mkdir(parents=True, exist_ok=True)
        sky_overlay_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device()
    print(f"Device: {device}")
    print(f"Found {len(hazy_images)} hazy images. Processing up to {MAX_IMAGES} matched pairs.")

    predictor = load_sam_predictor(device)

    cmdn = CMDN(
        dino_source_dir=DINO_SOURCE_DIR,
        dino_weight_path=DINO_WEIGHT_PATH,
    ).to(device)
    cmdn.eval()

    rows = []
    processed = 0

    with torch.no_grad():
        for hazy_path in hazy_images:
            if processed >= MAX_IMAGES:
                break

            ir_path = find_ir_image(hazy_path, ir_by_name, ir_by_stem)
            if ir_path is None:
                print(f"WARNING: no matching IR image for {hazy_path.name}; skipped.")
                continue

            x_vis = load_rgb(hazy_path).unsqueeze(0).to(device)
            x_ir = load_rgb(ir_path).unsqueeze(0).to(device)
            vis_01 = denorm_clip_for_display(x_vis)
            sky_prior = compute_sky_prior(vis_01)

            sky_mask_tensor = None
            if USE_SAM_SKY_MASK:
                try:
                    vis_uint8 = (np.clip(vis_01, 0.0, 1.0) * 255).astype(np.uint8)
                    sky_mask = generate_sam_sky_mask(predictor, vis_uint8, sky_prior)
                    save_sky_mask(sky_mask_dir / f"{hazy_path.stem}_sky.png", sky_mask)
                    save_sky_overlay(sky_overlay_dir / f"{hazy_path.stem}_overlay.png",
                                     vis_01, sky_mask)
                    sky_mask_tensor = torch.from_numpy(sky_mask).unsqueeze(0).unsqueeze(0).to(device)
                except Exception as exc:
                    print(f"WARNING: SAM sky_mask failed for {hazy_path.name}; skipped. {exc}")
                    continue

            debug0 = cmdn(
                x_vis,
                x_ir,
                disc_alpha=0.0,
                return_debug=True,
                sky_mask=sky_mask_tensor,
            )
            debug1 = cmdn(
                x_vis,
                x_ir,
                disc_alpha=1.0,
                return_debug=True,
                sky_mask=sky_mask_tensor,
            )
            validate_debug_keys(debug0, hazy_path.name)
            validate_debug_keys(debug1, hazy_path.name)

            row = stats_for_image(hazy_path.name, debug0, debug1)
            rows.append(row)
            print_stats(row)

            if single_png_out is not None and len(hazy_images) == 1:
                out_png = single_png_out
            else:
                out_png = vis_dir / f"{hazy_path.stem}_debug.png"
            save_debug_figure(out_png, hazy_path.name, x_vis, x_ir, debug0, sky_prior)
            print(f"Saved visualization: {out_png}")

            processed += 1

    if not rows:
        raise RuntimeError("No matched VIS/IR image pairs were processed.")

    csv_path = out_dir / "pseudo_stats.csv"
    summary_path = out_dir / "summary.txt"
    save_csv(csv_path, rows)
    save_summary(summary_path, rows)

    print("-" * 80)
    print(f"Saved CSV: {csv_path}")
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
