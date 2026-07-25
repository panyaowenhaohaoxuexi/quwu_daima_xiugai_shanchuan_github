"""Pure mask-aware scalar loss primitives."""

import torch
from torch.nn import functional as F


def _expanded(mask: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return mask.detach().expand_as(value).to(dtype=value.dtype)


def masked_mean(value: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Mean over a detached support; empty support stays attached zero."""
    weight = _expanded(mask, value)
    denominator = weight.sum()
    if float(denominator.detach()) == 0.0:
        return value.sum() * 0.0
    return (value * weight).sum() / denominator.clamp_min(eps)


def masked_smooth_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
                     beta: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    if beta <= 0:
        raise ValueError("beta must be > 0")
    error = (prediction - target).abs()
    value = torch.where(error < beta, 0.5 * error.square() / beta, error - 0.5 * beta)
    return masked_mean(value, mask, eps)


def masked_bce(probability: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
               weight: torch.Tensor | None = None, eps: float = 1e-6) -> torch.Tensor:
    probability = probability.clamp(eps, 1.0 - eps)
    target = target.detach()
    value = -(target * probability.log() + (1.0 - target) * (1.0 - probability).log())
    effective_mask = mask.detach()
    if weight is not None:
        effective_mask = effective_mask * weight.detach()
    return masked_mean(value, effective_mask, eps)


def weighted_l1(value: torch.Tensor, target: torch.Tensor, weight: torch.Tensor,
                eps: float = 1e-6) -> torch.Tensor:
    expanded = weight.detach().expand_as(value)
    return (expanded * (value - target.detach()).abs()).sum() / expanded.sum().clamp_min(eps)


def weighted_bce(probability: torch.Tensor, target: torch.Tensor, weight: torch.Tensor,
                 eps: float = 1e-6) -> torch.Tensor:
    probability = probability.clamp(eps, 1.0 - eps)
    target, expanded = target.detach(), weight.detach().expand_as(probability)
    error = -(target * probability.log() + (1.0 - target) * (1.0 - probability).log())
    return (expanded * error).sum() / expanded.sum().clamp_min(eps)


__all__ = ["masked_mean", "masked_smooth_l1", "masked_bce", "weighted_l1", "weighted_bce"]
