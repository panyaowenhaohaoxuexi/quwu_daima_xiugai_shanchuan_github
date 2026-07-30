"""Two-stage FLIR-to-M3FD unsupervised domain adaptation entry point."""

import torch

from training.target_style import apply_target_statistics


def route_consistency_multiplier(step, *, warmup_steps, ramp_steps):
    """Keep route self-consistency off before its target-domain warm-up ends."""
    if step < 0 or warmup_steps < 0 or ramp_steps < 0:
        raise ValueError("route consistency steps must be non-negative")
    if step <= warmup_steps:
        return 0.0
    if ramp_steps == 0:
        return 1.0
    return min(1.0, (step - warmup_steps) / float(ramp_steps))


def style_source_batch(source_batch, target_hazy, target_tir, *, generator, probability,
                       beta_min, beta_max, min_gain, max_gain, max_abs_bias):
    """Apply target statistics only to Source modalities, never to physical labels."""
    hazy, clear, tir, density_gt, completion_mask_gt = source_batch
    if target_hazy.shape[0] != hazy.shape[0] or target_tir.shape[0] != hazy.shape[0]:
        raise ValueError("target reference batch size must match the Source batch")
    if not 0.0 <= float(probability) <= 1.0 or not 0.0 <= float(beta_min) <= float(beta_max) <= 1.0:
        raise ValueError("invalid target style probability or beta range")
    apply_mask = torch.rand(hazy.shape[0], generator=generator, device="cpu") < float(probability)
    beta = torch.empty(hazy.shape[0], device=hazy.device, dtype=hazy.dtype).uniform_(
        float(beta_min), float(beta_max), generator=generator
    )
    beta = beta * apply_mask.to(device=hazy.device, dtype=hazy.dtype)
    styled = apply_target_statistics(
        hazy, clear, tir, target_hazy.to(device=hazy.device, dtype=hazy.dtype),
        target_tir.to(device=hazy.device, dtype=hazy.dtype), beta=beta,
        min_gain=min_gain, max_gain=max_gain, max_abs_bias=max_abs_bias,
    )
    return (styled.hazy, styled.clear, styled.tir, density_gt, completion_mask_gt), apply_mask.to(hazy.device)
