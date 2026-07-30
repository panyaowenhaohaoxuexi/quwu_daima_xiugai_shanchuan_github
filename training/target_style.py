"""Bounded target-domain photometric statistics for labeled Source batches."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StyledSourceBatch:
    """Transformed Source modalities plus the applied per-sample affine maps."""

    hazy: torch.Tensor
    clear: torch.Tensor
    tir: torch.Tensor
    rgb_gain: torch.Tensor
    rgb_bias: torch.Tensor
    tir_gain: torch.Tensor
    tir_bias: torch.Tensor


def _require_modal(name, value, *, batch_size=None):
    if not isinstance(value, torch.Tensor) or value.ndim != 4 or value.shape[1] != 3:
        raise ValueError(f"{name} must have shape [B,3,H,W]")
    if batch_size is not None and value.shape[0] != batch_size:
        raise ValueError(f"{name} batch size must match hazy")


def _batch_scalar(value, *, batch_size, device, dtype, name):
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.numel() == 1:
        tensor = tensor.expand(batch_size)
    if tensor.ndim != 1 or tensor.shape[0] != batch_size:
        raise ValueError(f"{name} must be scalar or shape [B]")
    return tensor.reshape(batch_size, 1, 1, 1)


def _mean_std(value):
    return value.mean(dim=(-2, -1), keepdim=True), value.std(dim=(-2, -1), keepdim=True, unbiased=False)


def _affine_from_statistics(source, target, beta, *, min_gain, max_gain, max_abs_bias):
    source_mean, source_std = _mean_std(source)
    target_mean, target_std = _mean_std(target)
    raw_gain = (target_std / source_std.clamp_min(1e-6)).clamp(min_gain, max_gain)
    gain = 1.0 + beta * (raw_gain - 1.0)
    blended_mean = source_mean + beta * (target_mean - source_mean)
    bias = (blended_mean - gain * source_mean).clamp(-max_abs_bias, max_abs_bias)
    return gain, bias


def apply_target_statistics(hazy, clear, tir, target_hazy, target_tir, *, beta,
                            min_gain=0.75, max_gain=1.35, max_abs_bias=0.20):
    """Stylize only Source modalities using bounded global M3FD-like statistics.

    The caller deliberately retains density and completion-mask labels unchanged.
    """
    _require_modal("hazy", hazy)
    batch_size = hazy.shape[0]
    for name, value in (("clear", clear), ("tir", tir), ("target_hazy", target_hazy), ("target_tir", target_tir)):
        _require_modal(name, value, batch_size=batch_size)
    if not (0 < min_gain <= max_gain and max_abs_bias >= 0):
        raise ValueError("invalid target-statistics affine bounds")
    beta = _batch_scalar(beta, batch_size=batch_size, device=hazy.device, dtype=hazy.dtype, name="beta")
    if (beta < 0).any() or (beta > 1).any():
        raise ValueError("beta must be in [0, 1]")

    rgb_identity_gain = torch.ones((batch_size, 3, 1, 1), device=hazy.device, dtype=hazy.dtype)
    rgb_identity_bias = torch.zeros_like(rgb_identity_gain)
    tir_identity_gain = torch.ones((batch_size, 1, 1, 1), device=hazy.device, dtype=hazy.dtype)
    tir_identity_bias = torch.zeros_like(tir_identity_gain)
    if torch.equal(beta, torch.zeros_like(beta)):
        return StyledSourceBatch(hazy, clear, tir, rgb_identity_gain, rgb_identity_bias,
                                 tir_identity_gain, tir_identity_bias)

    rgb_gain, rgb_bias = _affine_from_statistics(
        hazy, target_hazy, beta, min_gain=min_gain, max_gain=max_gain, max_abs_bias=max_abs_bias
    )
    tir_gain, tir_bias = _affine_from_statistics(
        tir[:, :1], target_tir[:, :1], beta, min_gain=min_gain, max_gain=max_gain,
        max_abs_bias=max_abs_bias
    )
    return StyledSourceBatch(
        (rgb_gain * hazy + rgb_bias).clamp(0.0, 1.0),
        (rgb_gain * clear + rgb_bias).clamp(0.0, 1.0),
        (tir_gain * tir + tir_bias).clamp(0.0, 1.0),
        rgb_gain,
        rgb_bias,
        tir_gain,
        tir_bias,
    )
