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


def _id_map_to_rgb(id_map, valid_mask=None, num_ids=32):
    if id_map.dim() == 4:
        id_map = id_map.squeeze(0)
    if id_map.dim() == 3:
        id_map = id_map.squeeze(0)
    id_map = id_map.detach().cpu().long()
    palette = torch.tensor(
        [
            [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
            [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
            [210, 245, 60], [250, 190, 190], [0, 128, 128], [230, 190, 255],
            [170, 110, 40], [255, 250, 200], [128, 0, 0], [170, 255, 195],
            [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128],
        ],
        dtype=torch.float32,
    ) / 255.0
    colors = palette[id_map.remainder(palette.shape[0])]
    if valid_mask is not None:
        if valid_mask.dim() == 4:
            valid_mask = valid_mask.squeeze(0)
        if valid_mask.dim() == 3:
            valid_mask = valid_mask.squeeze(0)
        colors = colors * valid_mask.detach().cpu().float().unsqueeze(-1)
    return colors.permute(2, 0, 1).clamp(0.0, 1.0)


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
    proto_assign=None,
    proto_attn=None,
    max_sim_map=None,
    max_samples=4,
):
    panel_size = 192
    title_h = 32
    row_label_w = 90
    padding = 6
    n = min(max_samples, hazy_vis.shape[0], 4)
    columns = list(_COLUMNS)
    if proto_assign is not None:
        columns.append("Proto_assign")
    if proto_attn is not None:
        columns.append("Proto_attn_top1")
    if max_sim_map is not None:
        columns.append("IR_proto_sim")
    num_cols = len(columns)
    width = row_label_w + num_cols * panel_size
    height = title_h + n * panel_size
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for col, title in enumerate(columns):
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
        if proto_assign is not None:
            assign = proto_assign[row:row + 1]
            assign_top1 = assign.argmax(dim=1, keepdim=True).float()
            reliable = 1.0 - F.interpolate(binary_mask[row:row + 1].float(), size=assign.shape[-2:], mode='nearest')
            panels.append(_resize_panel(_id_map_to_rgb(assign_top1, reliable, assign.shape[1]).unsqueeze(0), panel_size))
        if proto_attn is not None:
            attn = proto_attn[row:row + 1]
            n = attn.shape[1]
            side_h = proto_assign.shape[-2] if proto_assign is not None else int(n ** 0.5)
            side_w = proto_assign.shape[-1] if proto_assign is not None else max(1, n // max(1, side_h))
            attn_top1 = attn.argmax(dim=-1).view(1, 1, side_h, side_w).float()
            completion = F.interpolate(binary_mask[row:row + 1].float(), size=(side_h, side_w), mode='nearest')
            panels.append(_resize_panel(_id_map_to_rgb(attn_top1, completion, attn.shape[-1]).unsqueeze(0), panel_size))
        if max_sim_map is not None:
            sim = max_sim_map[row:row + 1]
            sim_min = sim.amin(dim=(2, 3), keepdim=True)
            sim_max = sim.amax(dim=(2, 3), keepdim=True)
            sim_norm = (sim - sim_min) / (sim_max - sim_min + 1e-6)
            panels.append(_resize_panel(sim_norm, panel_size))
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
