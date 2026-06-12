"""
Offline SAM sky-mask generation.

This script only generates precomputed SAM sky masks and overlay previews.
It does not train a model, modify model weights, or affect CMDN directly.
The training stage never runs SAM online; after generation, set
train_sky_mask_dir in option/Teacher.py and train with --use_train_sky_mask.
To generate masks for real_test_specific_hazy_dir, edit HAZY_DIR,
OUT_MASK_DIR, and OUT_OVERLAY_DIR below and run this script again.
"""

import os
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


HAZY_DIR = r"/root/autodl-tmp/FLIR_zengqiang/train/hazy"
OUT_MASK_DIR = r"/root/autodl-tmp/FLIR_zengqiang/train/sky_mask"
OUT_OVERLAY_DIR = r"/root/autodl-tmp/FLIR_zengqiang/train/sky_overlay"

SAM_CHECKPOINT = r"./sam_model/sam_vit_h_4b8939.pth"
SAM_MODEL_TYPE = "vit_h"

DEVICE = "cuda"

IMAGE_EXTENSIONS = [".jpg", ".png", ".jpeg"]
MAX_IMAGES = -1

SAM_INPUT_SIZE = 512

SKY_MASK_SUFFIX = "_sky"
SKY_MASK_EXT = ".png"

SKY_POS_POINT_NUM = 5
SKY_NEG_POINT_NUM = 5
SKY_PRIOR_TOP_RATIO = 0.55
SKY_MASK_THRESHOLD = 0.5

SKY_MASK_TOO_LARGE = 0.65
SKY_MASK_TOO_SMALL = 0.01

OVERWRITE_EXISTING = False


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))


def to_numpy_rgb01(image):
    arr = np.asarray(image).astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def compute_sky_prior(vis_01):
    vis = np.asarray(vis_01, dtype=np.float32)
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
    point_labels = np.array([1] * len(pos_points) + [0] * len(neg_points),
                            dtype=np.int32)
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
        prior_inside = float(sky_prior[mask_bool].mean()) if mask_bool.any() else 0.0
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
    vis = np.asarray(vis_01, dtype=np.float32)
    mask = np.clip(sky_mask[..., None], 0.0, 1.0)
    color = np.array([0.1, 0.65, 1.0], dtype=np.float32)
    overlay = vis * (1.0 - 0.45 * mask) + color * (0.45 * mask)
    Image.fromarray((np.clip(overlay, 0.0, 1.0) * 255).astype(np.uint8)).save(path)


def collect_images(hazy_dir):
    exts = {ext.lower() for ext in IMAGE_EXTENSIONS}
    images = [p for p in Path(hazy_dir).iterdir()
              if p.is_file() and p.suffix.lower() in exts]
    images = sorted(images, key=lambda p: p.name)
    if MAX_IMAGES > 0:
        images = images[:MAX_IMAGES]
    if not images:
        raise RuntimeError(f"No images found in {hazy_dir} with extensions {IMAGE_EXTENSIONS}")
    return images


def main():
    hazy_dir = Path(HAZY_DIR)
    out_mask_dir = Path(OUT_MASK_DIR)
    out_overlay_dir = Path(OUT_OVERLAY_DIR)
    sam_checkpoint = Path(SAM_CHECKPOINT)

    if not hazy_dir.is_dir():
        raise RuntimeError(f"HAZY_DIR does not exist: {hazy_dir}")
    if not sam_checkpoint.is_file():
        raise RuntimeError(f"SAM_CHECKPOINT does not exist: {sam_checkpoint}")

    try:
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError as exc:
        raise ImportError("segment_anything is required to generate SAM sky masks.") from exc

    out_mask_dir.mkdir(parents=True, exist_ok=True)
    out_overlay_dir.mkdir(parents=True, exist_ok=True)

    device = DEVICE
    if device == "cuda" and not torch.cuda.is_available():
        print("WARNING: DEVICE='cuda' but cuda is not available. Falling back to cpu.")
        device = "cpu"

    images = collect_images(hazy_dir)
    print(f"Found {len(images)} image(s).")
    print(f"Loading SAM: type={SAM_MODEL_TYPE}, checkpoint={sam_checkpoint}, device={device}")

    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(sam_checkpoint))
    sam.to(device)
    sam.eval()
    predictor = SamPredictor(sam)

    generated_count = 0
    skipped_existing_count = 0
    failed_count = 0
    mask_areas = []

    with torch.no_grad():
        for idx, image_path in enumerate(images, 1):
            mask_path = out_mask_dir / f"{image_path.stem}{SKY_MASK_SUFFIX}{SKY_MASK_EXT}"
            overlay_path = out_overlay_dir / f"{image_path.stem}_overlay.png"
            if mask_path.exists() and not OVERWRITE_EXISTING:
                print(f"[{idx}/{len(images)}] skip existing: {mask_path.name}")
                skipped_existing_count += 1
                continue

            try:
                image = Image.open(image_path).convert("RGB")
                original_size = image.size
                sam_image = image
                if SAM_INPUT_SIZE and max(original_size) != SAM_INPUT_SIZE:
                    scale = float(SAM_INPUT_SIZE) / float(max(original_size))
                    resize_size = (
                        max(1, int(round(original_size[0] * scale))),
                        max(1, int(round(original_size[1] * scale))),
                    )
                    sam_image = image.resize(resize_size, Image.BICUBIC)

                vis_01_sam = to_numpy_rgb01(sam_image)
                vis_uint8_sam = (vis_01_sam * 255).astype(np.uint8)
                sky_prior = compute_sky_prior(vis_01_sam)
                sky_mask_sam = generate_sam_sky_mask(predictor, vis_uint8_sam, sky_prior)

                sky_mask_img = Image.fromarray((sky_mask_sam * 255).astype(np.uint8))
                if sky_mask_img.size != original_size:
                    sky_mask_img = sky_mask_img.resize(original_size, Image.NEAREST)
                sky_mask = (np.asarray(sky_mask_img).astype(np.float32) / 255.0 >= 0.5).astype(np.float32)

                save_sky_mask(mask_path, sky_mask)
                save_sky_overlay(overlay_path, to_numpy_rgb01(image), sky_mask)

                sky_mask_mean = float(sky_mask.mean())
                mask_areas.append(sky_mask_mean)
                if sky_mask_mean > SKY_MASK_TOO_LARGE:
                    print(f"WARNING: sky_mask may be too large: {image_path.name}")
                if sky_mask_mean < SKY_MASK_TOO_SMALL:
                    print(f"WARNING: sky_mask may be too small: {image_path.name}")
                print(f"[{idx}/{len(images)}] generated: {image_path.name}, area={sky_mask_mean:.6f}")
                generated_count += 1
            except Exception as exc:
                failed_count += 1
                print(f"WARNING: failed to generate sky mask for {image_path.name}: {exc}")

    mean_area = float(np.mean(mask_areas)) if mask_areas else 0.0
    summary_path = out_mask_dir / "sky_mask_generation_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"total_images={len(images)}\n")
        f.write(f"generated_count={generated_count}\n")
        f.write(f"skipped_existing_count={skipped_existing_count}\n")
        f.write(f"failed_count={failed_count}\n")
        f.write(f"mean_sky_mask_area={mean_area:.10f}\n")
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
