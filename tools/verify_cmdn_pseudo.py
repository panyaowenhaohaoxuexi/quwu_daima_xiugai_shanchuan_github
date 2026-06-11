"""
Offline verification for CMDN fixed pseudo-label P_pseudo.

Edit HAZY_DIR, IR_DIR, OUT_DIR, SIZE, and MAX_IMAGES below, then run:

python tools/verify_cmdn_pseudo.py

HAZY_DIR and IR_DIR may be either folders or matching single image files.
OUT_DIR may be a folder, or a .png path for a single debug visualization.

This script does not train, create an optimizer, compute losses, call
backward, or load the Teacher/VIFNet model. It only runs CMDN with
return_debug=True and saves pseudo-label diagnostics.
"""

import csv
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# User-editable configuration
# ---------------------------------------------------------------------------

HAZY_DIR = r"F:/Dehaze_Paper/2_Dataset/1_main_benchmark/REAL_FOGGY/hazy/01435.png"
IR_DIR = r"F:/Dehaze_Paper/2_Dataset/1_main_benchmark/REAL_FOGGY/ir/01435.png"
OUT_DIR = r"D:/liu_lan_qi_xia_zai/quwu_daima_xiugai_shanchuan_github/cmdn_verify_results/01435_pseudo.png"

SIZE = 512
MAX_IMAGES = 20
DEVICE = "cuda"

DINO_SOURCE_DIR = r"./DINOv2/facebookresearch_dinov2_main"
DINO_WEIGHT_PATH = r"./dinov2_model/dinov2_vitb14_pretrain.pth"

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


def plot_heat(ax, data, title, cmap="hot"):
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def save_debug_figure(out_path, filename, x_vis, x_ir, debug):
    vis_01 = denorm_clip_for_display(x_vis)
    ir_01 = denorm_clip_for_display(x_ir)

    maps = {
        "g_fog diagnostic": map_to_numpy(debug["g_fog"]),
        "haze_app": map_to_numpy(debug["haze_app"]),
        "attn_deg": map_to_numpy(debug["attn_deg"]),
        "struct_deg": map_to_numpy(debug["struct_deg"]),
        "ir_adv": map_to_numpy(debug["ir_adv"]),
        "P_pseudo": map_to_numpy(debug["P_pseudo"]),
        "P_fail": map_to_numpy(debug["P_fail"]),
        "G_dec": map_to_numpy(debug["G_dec"]),
        "P_support": map_to_numpy(debug["P_support"]),
        "G_soft": map_to_numpy(debug["G_soft"]),
        "M_hard": map_to_numpy(debug["M_hard"]),
    }

    p_fail_map = maps["P_fail"]
    tau_value = scalar_mean(debug["tau"])
    tau_map = np.full_like(p_fail_map, tau_value, dtype=np.float32)

    fig, axes = plt.subplots(1, 14, figsize=(48, 4), squeeze=False)
    axes = axes[0]

    axes[0].imshow(vis_01)
    axes[0].set_title("Hazy Visible", fontsize=8)
    axes[0].axis("off")

    axes[1].imshow(ir_01)
    axes[1].set_title("Infrared", fontsize=8)
    axes[1].axis("off")

    plot_heat(axes[2], maps["g_fog diagnostic"], "g_fog diagnostic")
    plot_heat(axes[3], maps["haze_app"], "haze_app")
    plot_heat(axes[4], maps["attn_deg"], "attn_deg")
    plot_heat(axes[5], maps["struct_deg"], "struct_deg")
    plot_heat(axes[6], maps["ir_adv"], "ir_adv")
    plot_heat(axes[7], maps["P_pseudo"], "P_pseudo")
    plot_heat(axes[8], maps["P_fail"], "P_fail")
    plot_heat(axes[9], tau_map, "tau", cmap="viridis")
    plot_heat(axes[10], maps["G_dec"], "G_dec")
    plot_heat(axes[11], maps["P_support"], "P_support")
    plot_heat(axes[12], maps["G_soft"], "G_soft")
    plot_heat(axes[13], maps["M_hard"], "M_hard", cmap="gray")

    fig.suptitle(filename, fontsize=11, y=1.05)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def stats_for_image(filename, debug0, debug1):
    p = debug0["P_pseudo"]
    max_abs_alpha_diff = (debug0["P_pseudo"] - debug1["P_pseudo"]).abs().max().item()

    if max_abs_alpha_diff > 1e-6:
        print("WARNING: P_pseudo changes with disc_alpha; fixed pseudo-label assumption is violated.")

    row = {
        "filename": filename,
        "p_min": float(p.detach().cpu().float().min().item()),
        "p_mean": scalar_mean(p),
        "p_max": float(p.detach().cpu().float().max().item()),
        "m_hard_mean": scalar_mean(debug0["M_hard"]),
        "g_soft_mean": scalar_mean(debug0["G_soft"]),
        "p_fail_mean": scalar_mean(debug0["P_fail"]),
        "tau_mean": scalar_mean(debug0["tau"]),
        "max_abs_alpha_diff": float(max_abs_alpha_diff),
        "requires_grad": bool(p.requires_grad),
    }
    return row


def print_stats(row):
    print("-" * 80)
    print(f"filename: {row['filename']}")
    print(
        "P_pseudo min/mean/max: "
        f"{row['p_min']:.6f} / {row['p_mean']:.6f} / {row['p_max']:.6f}")
    print(f"M_hard mean: {row['m_hard_mean']:.6f}")
    print(f"G_soft mean: {row['g_soft_mean']:.6f}")
    print(f"P_fail mean: {row['p_fail_mean']:.6f}")
    print(f"tau mean: {row['tau_mean']:.6f}")
    print(f"disc_alpha stability max_abs: {row['max_abs_alpha_diff']:.10f}")
    print(f"P_pseudo.requires_grad: {row['requires_grad']}")


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

    lines = [
        f"num_images={len(rows)}",
        f"mean_pseudo_mean={float(np.mean(pseudo_means)):.10f}",
        f"mean_pseudo_max={float(np.mean(pseudo_maxes)):.10f}",
        f"mean_m_hard_mean={float(np.mean(m_hard_means)):.10f}",
        f"mean_alpha_diff={float(np.mean(alpha_diffs)):.10f}",
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

    ensure_input_paths(hazy_dir, ir_dir)
    hazy_images = collect_hazy_images(hazy_dir)
    ir_by_name, ir_by_stem = build_ir_index(ir_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device()
    print(f"Device: {device}")
    print(f"Found {len(hazy_images)} hazy images. Processing up to {MAX_IMAGES} matched pairs.")

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

            debug0 = cmdn(x_vis, x_ir, disc_alpha=0.0, return_debug=True)
            debug1 = cmdn(x_vis, x_ir, disc_alpha=1.0, return_debug=True)
            validate_debug_keys(debug0, hazy_path.name)
            validate_debug_keys(debug1, hazy_path.name)

            row = stats_for_image(hazy_path.name, debug0, debug1)
            rows.append(row)
            print_stats(row)

            if single_png_out is not None and len(hazy_images) == 1:
                out_png = single_png_out
            else:
                out_png = vis_dir / f"{hazy_path.stem}_debug.png"
            save_debug_figure(out_png, hazy_path.name, x_vis, x_ir, debug0)
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
