"""Synthetic density, route and boundary loss formulas."""

import torch

from loss.common.masked import masked_bce, masked_mean, masked_smooth_l1
from loss.common.structural import masked_gradient_error


def binary_route_penalty(route_soft: torch.Tensor, validity_mask: torch.Tensor,
                         eps: float = 1e-6) -> torch.Tensor:
    return masked_mean(4.0 * route_soft * (1.0 - route_soft), validity_mask, eps)


def density_loss(density_map: torch.Tensor, density_gt: torch.Tensor, validity: torch.Tensor,
                 beta: float) -> torch.Tensor:
    return masked_smooth_l1(density_map, density_gt, validity, beta)


def route_supervision_loss(route_soft: torch.Tensor, q: torch.Tensor, omega_support: torch.Tensor,
                           omega_weight: torch.Tensor | None = None) -> torch.Tensor:
    return masked_bce(route_soft, q.detach(), omega_support.detach(), omega_weight)


def boundary_loss(prediction: torch.Tensor, target: torch.Tensor, boundary_map: torch.Tensor,
                  validity: torch.Tensor, *, l1_weight: float,
                  gradient_weight: float) -> torch.Tensor:
    support = boundary_map.detach() * validity
    return (l1_weight * masked_mean((prediction - target).abs(), support) +
            gradient_weight * masked_gradient_error(prediction, target, support))


__all__ = ["binary_route_penalty", "density_loss", "route_supervision_loss", "boundary_loss"]
