import os

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
_COLUMNS = [
    "Hazy",
    "IR",
    "Pred",
    "Clear",
    "Density_pred",
    "Density_gt",
    "Mask_prob",
    "Binary_mask",
    "Mask_gt",
]


def _denorm_clip(x):
    mean = _CLIP_MEAN.to(device=x.device, dtype=x.dtype)
    std = _CLIP_STD.to(device=x.device, dtype=x.dtype)
    return (x * std + mean).clamp(0.0, 1.0)


def tensor_to_pil_img(x):
    x = x.detach().cpu().clamp(0, 1)
    x = (x * 255).byte()
    return Image.fromarray(x.permute(1, 2, 0).numpy())


def _resize_panel(x, panel_size, binary=False):
    if x.dim() == 3:
        x = x.unsqueeze(0)
    if binary:
        x = (x >= 0.5).float()
        x = F.interpolate(x, size=(panel_size, panel_size), mode='nearest')
    else:
        x = F.interpolate(x, size=(panel_size, panel_size), mode='bilinear', align_corners=False)
    x = x.clamp(0.0, 1.0)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    elif x.shape[1] != 3:
        x = x[:, :1].repeat(1, 3, 1, 1)
    return x.squeeze(0)


def _text_size(draw, text, font):
    if hasattr(draw, "textbbox"):
        box = draw.textbbox((0, 0), text, font=font)
        return box[2] - box[0], box[3] - box[1]
    return draw.textsize(text, font=font)


def _draw_centered_text(draw, box, text, font, fill=(20, 20, 20)):
    x0, y0, x1, y1 = box
    text_w, text_h = _text_size(draw, text, font)
    x = x0 + max(0, (x1 - x0 - text_w) // 2)
    y = y0 + max(0, (y1 - y0 - text_h) // 2)
    draw.text((x, y), text, fill=fill, font=font)


def save_teacher_region_visualization(
    save_dir,
    prefix,
    hazy_vis,
    infrared,
    pred_clear,
    clear_vis,
    density_map,
    density_gt,
    mask_prob,
    binary_mask,
    mask_gt,
    max_samples=4,
):
    panel_size = 192
    title_h = 32
    row_label_w = 90
    padding = 6
    n = min(max_samples, hazy_vis.shape[0], 4)
    num_cols = len(_COLUMNS)
    width = row_label_w + num_cols * panel_size
    height = title_h + n * panel_size
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for col, title in enumerate(_COLUMNS):
        x0 = row_label_w + col * panel_size
        _draw_centered_text(draw, (x0, 0, x0 + panel_size, title_h), title, font)

    for row in range(n):
        y0 = title_h + row * panel_size
        _draw_centered_text(draw, (0, y0, row_label_w, y0 + panel_size), f"Scene {row + 1}", font)
        panels = [
            _resize_panel(_denorm_clip(hazy_vis[row:row + 1]), panel_size),
            _resize_panel(_denorm_clip(infrared[row:row + 1]), panel_size),
            _resize_panel(pred_clear[row:row + 1], panel_size),
            _resize_panel(clear_vis[row:row + 1], panel_size),
            _resize_panel(density_map[row:row + 1], panel_size),
            _resize_panel(density_gt[row:row + 1], panel_size),
            _resize_panel(mask_prob[row:row + 1], panel_size),
            _resize_panel(binary_mask[row:row + 1], panel_size, binary=True),
            _resize_panel(mask_gt[row:row + 1], panel_size, binary=True),
        ]
        for col, panel in enumerate(panels):
            x0 = row_label_w + col * panel_size
            image = tensor_to_pil_img(panel)
            if padding > 0:
                resample = Image.NEAREST if col in (7, 8) else Image.BILINEAR
                image = image.resize((panel_size - 2 * padding, panel_size - 2 * padding), resample)
            canvas.paste(image, (x0 + padding, y0 + padding))

    out_dir = os.path.join(save_dir, "teacher_region_vis", prefix)
    os.makedirs(out_dir, exist_ok=True)
    canvas.save(os.path.join(out_dir, "overview.png"))
