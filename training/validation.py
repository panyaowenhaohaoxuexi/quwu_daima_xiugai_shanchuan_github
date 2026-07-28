"""Paired RGB--TIR validation and best-checkpoint selection."""

from pathlib import Path

import torch

from utils.metrics import psnr, ssim_global


@torch.inference_mode()
def evaluate_paired_validation(model, loader, device, *, route_temperature):
    """Evaluate dehazed RGB against paired clear targets at original resolution."""
    was_training = model.training
    model.eval()
    psnr_values, ssim_values = [], []
    for hazy, clear, tir, _density in loader:
        if hazy.numel() == 0:
            continue
        output = model(hazy.to(device), tir.to(device), route_temperature=route_temperature, route_mode="hard")
        prediction, target = output["pred_clear"], clear.to(device)
        psnr_values.append(psnr(prediction, target))
        ssim_values.append(ssim_global(prediction, target))
    if was_training:
        model.train()
    if not psnr_values:
        raise ValueError("validation loader yielded no valid paired samples")
    return {"psnr": float(torch.stack(psnr_values).mean().cpu()),
            "ssim": float(torch.stack(ssim_values).mean().cpu())}


def save_best_if_improved(validation_psnr, best_psnr, checkpoint, path):
    """Persist a candidate checkpoint only when its PSNR is strictly better."""
    if validation_psnr > best_psnr:
        torch.save(checkpoint, Path(path))
        return float(validation_psnr)
    return float(best_psnr)
