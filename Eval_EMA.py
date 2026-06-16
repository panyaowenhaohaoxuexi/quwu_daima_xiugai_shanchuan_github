#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import os
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from PIL import Image, ImageDraw
from tqdm import tqdm
from torchvision.transforms import Compose, Normalize, ToTensor, Resize, InterpolationMode

from model import VIFNetInconsistencyTeacher


MODEL_PATH = "/root/autodl-tmp/CoA-main_daima_xiugai_teacher_v6/Teacher_xunlian/saved_model/best.pth"
OUTPUT_FOLDER = "/root/autodl-tmp/CoA-main_daima_xiugai_teacher_v10/v6_code_xiugai_test/Teacher_model_test_v4"
INPUT_FOLDER_VIS = "/root/autodl-tmp/REAL_FOGGY_autodl/hazy"
INPUT_FOLDER_IR = "/root/autodl-tmp/REAL_FOGGY_autodl/ir"

SAVE_INTERNAL_OVERVIEW = True
EVAL_OVERVIEW_COLUMNS = [
    "Hazy",
    "IR",
    "Pred",
    "Density_pred",
    "Mask_prob",
    "Binary_mask",
]

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
MODEL_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
MODEL_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)

transform = Compose([
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])


def denormalize_model_input(tensor):
    mean = MODEL_MEAN.to(device=tensor.device, dtype=tensor.dtype)
    std = MODEL_STD.to(device=tensor.device, dtype=tensor.dtype)
    return (tensor * std + mean).clamp(0, 1)


def find_paired_image(folder, stem):
    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(folder, stem + ext)
        if os.path.exists(candidate):
            return candidate
    matches = sorted(glob.glob(os.path.join(folder, stem + ".*")))
    for candidate in matches:
        if os.path.splitext(candidate)[1].lower() in IMAGE_EXTENSIONS:
            return candidate
    return None


def resize_to_original(tensor, size, mode):
    if tuple(tensor.shape[-2:]) == tuple(size):
        return tensor
    if mode == "nearest":
        return F.interpolate(tensor, size=size, mode=mode)
    return F.interpolate(tensor, size=size, mode=mode, align_corners=False)


def tensor_to_pil(panel):
    panel = panel.detach().cpu().clamp(0, 1)
    if panel.dim() == 4:
        panel = panel.squeeze(0)
    if panel.size(0) == 1:
        panel = panel.repeat(3, 1, 1)
    return torchvision.transforms.functional.to_pil_image(panel)


def save_internal_overview(hazy, ir, pred, density, prob, binary, save_path):
    panels = [hazy, ir, pred, density, prob, binary]
    images = [tensor_to_pil(panel) for panel in panels]
    tile_w, tile_h = images[0].size
    label_h = 24
    canvas = Image.new("RGB", (tile_w * len(images), tile_h + label_h), "white")
    draw = ImageDraw.Draw(canvas)

    for idx, (label, image) in enumerate(zip(EVAL_OVERVIEW_COLUMNS, images)):
        x = idx * tile_w
        canvas.paste(image, (x, label_h))
        draw.text((x + 4, 4), label, fill=(0, 0, 0))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    canvas.save(save_path)


def dehaze(model, vis_image_path, ir_image_path, folder):
    try:
        haze_vis_pil = Image.open(vis_image_path).convert("RGB")
        haze_ir_pil = Image.open(ir_image_path).convert("RGB")

        haze_vis = transform(haze_vis_pil).unsqueeze(0).to(device)
        haze_ir = transform(haze_ir_pil).unsqueeze(0).to(device)
        h, w = haze_vis.shape[2], haze_vis.shape[3]

        target_h = max((h // 16) * 16, 16)
        target_w = max((w // 16) * 16, 16)
        resize_fn = Resize((target_h, target_w), interpolation=InterpolationMode.BICUBIC, antialias=True)
        haze_vis_resized = resize_fn(haze_vis)
        haze_ir_resized = resize_fn(haze_ir)

        with torch.no_grad():
            out = model(haze_vis_resized, haze_ir_resized, return_dict=True)

        pred_clear = out["pred_clear"]
        density_map = out["density_map"]
        mask_prob = out["mask_prob"]
        binary_mask = out["binary_mask"]

        original_size = (h, w)
        pred_clear_restored = resize_to_original(pred_clear, original_size, "bicubic")
        density_restored = resize_to_original(density_map, original_size, "bilinear")
        mask_prob_restored = resize_to_original(mask_prob, original_size, "bilinear")
        binary_mask_thresholded = (binary_mask >= 0.5).float()
        binary_mask_restored = resize_to_original(binary_mask_thresholded, original_size, "nearest")

        output_filename = os.path.basename(vis_image_path)
        stem = os.path.splitext(output_filename)[0]
        save_path = os.path.join(folder, output_filename)
        os.makedirs(folder, exist_ok=True)

        pred_to_save = pred_clear_restored.squeeze(0).clamp(0, 1)
        torchvision.utils.save_image(pred_to_save, save_path)
        print(f"[Eval] saved pred_clear: {save_path}")

        if SAVE_INTERNAL_OVERVIEW:
            overview_path = os.path.join(folder, "internal_mask_vis", f"{stem}_overview.png")
            save_internal_overview(
                denormalize_model_input(haze_vis),
                denormalize_model_input(haze_ir),
                pred_clear_restored.clamp(0, 1),
                density_restored.clamp(0, 1),
                mask_prob_restored.clamp(0, 1),
                binary_mask_restored.clamp(0, 1),
                overview_path,
            )
            print(f"[Eval] saved internal overview: {overview_path}")

    except FileNotFoundError as exc:
        print(f"[Eval] warning: missing image file {exc}; skip.")
    except Exception as exc:
        base_name = os.path.basename(vis_image_path)
        print(f"[Eval] warning: failed to process {base_name}: {exc}; skip.")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VIFNetInconsistencyTeacher().to(device)
    print("[Eval] loading model: VIFNetInconsistencyTeacher")

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        has_module_prefix = any(key.startswith("module.") for key in checkpoint.keys())
        state_dict = OrderedDict()
        for key, value in checkpoint.items():
            name = key[7:] if has_module_prefix else key
            state_dict[name] = value
        load_result = model.load_state_dict(state_dict, strict=True)
        print(f"[Eval] model load result: {load_result}")
    except FileNotFoundError:
        print(f"[Eval] error: model checkpoint not found: {MODEL_PATH}")
        raise SystemExit(1)
    except Exception as exc:
        print(f"[Eval] error: failed to load checkpoint: {exc}")
        print("[Eval] hint: inspect missing/unexpected keys or try strict=False manually.")
        raise SystemExit(1)

    model.eval()
    print("[Eval] using model-internal HDE + Gumbel binary_mask; no external mask post-processing.")

    if not os.path.isdir(INPUT_FOLDER_VIS):
        print(f"[Eval] error: visible input folder not found: {INPUT_FOLDER_VIS}")
        raise SystemExit(1)
    if not os.path.isdir(INPUT_FOLDER_IR):
        print(f"[Eval] error: infrared input folder not found: {INPUT_FOLDER_IR}")
        raise SystemExit(1)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    vis_images = []
    for ext in IMAGE_EXTENSIONS:
        vis_images.extend(glob.glob(os.path.join(INPUT_FOLDER_VIS, "*" + ext)))
    vis_images = sorted(vis_images)

    if not vis_images:
        print(f"[Eval] error: no images found in {INPUT_FOLDER_VIS}")
        raise SystemExit(1)

    bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} | Elapsed: {elapsed} | Rate: {rate_fmt} items/sec"
    for vis_path in tqdm(vis_images, bar_format=bar_format, desc="Internal-mask eval"):
        base_filename = os.path.basename(vis_path)
        stem = os.path.splitext(base_filename)[0]
        ir_path = find_paired_image(INPUT_FOLDER_IR, stem)
        if ir_path is None:
            print(f"[Eval] warning: no paired IR image for {base_filename}; skip.")
            continue
        dehaze(model, vis_path, ir_path, OUTPUT_FOLDER)

    print(f"[Eval] finished. pred_clear images saved to: {OUTPUT_FOLDER}")
