import os

import torch
import torchvision


_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
_CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def _denorm_clip(x):
    mean = _CLIP_MEAN.to(device=x.device, dtype=x.dtype)
    std = _CLIP_STD.to(device=x.device, dtype=x.dtype)
    return (x * std + mean).clamp(0.0, 1.0)


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
    os.makedirs(save_dir, exist_ok=True)
    n = min(max_samples, hazy_vis.shape[0])
    rows = [
        _denorm_clip(hazy_vis[:n]),
        _denorm_clip(infrared[:n]),
        pred_clear[:n].clamp(0.0, 1.0),
        clear_vis[:n].clamp(0.0, 1.0),
        density_map[:n].repeat(1, 3, 1, 1).clamp(0.0, 1.0),
        density_gt[:n].repeat(1, 3, 1, 1).clamp(0.0, 1.0),
        mask_prob[:n].repeat(1, 3, 1, 1).clamp(0.0, 1.0),
        binary_mask[:n].repeat(1, 3, 1, 1).clamp(0.0, 1.0),
        mask_gt[:n].repeat(1, 3, 1, 1).clamp(0.0, 1.0),
    ]
    grid = torchvision.utils.make_grid(torch.cat(rows, dim=0), nrow=n)
    torchvision.utils.save_image(grid, os.path.join(save_dir, f"{prefix}_teacher_region.png"))
