"""Complete synthetic source-domain objective."""

import torch

from .reconstruction import reconstruction_error
from .routing import binary_route_penalty, boundary_loss, density_loss, route_supervision_loss


def compute_source_objective(pred_clear: torch.Tensor, clear_rgb: torch.Tensor, density_map: torch.Tensor,
                             density_gt: torch.Tensor, route_soft: torch.Tensor, boundary_map: torch.Tensor,
                             q: torch.Tensor, omega_support: torch.Tensor, omega_weight: torch.Tensor | None = None,
                             density_beta: float = 0.1, lambda_global: float = 1.0, lambda_fuse: float = 1.0,
                             lambda_comp: float = 1.0, lambda_boundary: float = 1.0, lambda_density: float = 1.0,
                             lambda_route: float = 1.0, lambda_binary: float = 1.0, lambda_router: float = 1.0,
                             rec_l1_weight: float = 1.0, rec_gradient_weight: float = 0.0,
                             rec_ssim_weight: float = 0.0, boundary_l1_weight: float = 1.0,
                             boundary_gradient_weight: float = 0.0, ssim_window: int = 7,
                             min_valid_support: int = 4) -> dict[str, torch.Tensor]:
    validity = density_gt.new_ones(density_gt.shape)
    route, boundary = route_soft.detach(), boundary_map.detach()
    reconstruction_kwargs = {
        "l1_weight": rec_l1_weight, "gradient_weight": rec_gradient_weight,
        "ssim_weight": rec_ssim_weight, "ssim_window": ssim_window,
        "min_valid_support": min_valid_support,
    }
    global_loss = reconstruction_error(pred_clear, clear_rgb, validity, **reconstruction_kwargs)
    fuse_loss = reconstruction_error(pred_clear, clear_rgb, (1 - route) * validity, **reconstruction_kwargs)
    comp_loss = reconstruction_error(pred_clear, clear_rgb, route * validity, **reconstruction_kwargs)
    boundary_value = boundary_loss(pred_clear, clear_rgb, boundary, validity,
                                   l1_weight=boundary_l1_weight, gradient_weight=boundary_gradient_weight)
    density_value = density_loss(density_map, density_gt, validity, density_beta)
    route_value = route_supervision_loss(route_soft, q, omega_support, omega_weight)
    binary_value = binary_route_penalty(route_soft, validity)
    router_total = lambda_density * density_value + lambda_route * route_value + lambda_binary * binary_value
    total = (lambda_global * global_loss + lambda_fuse * fuse_loss + lambda_comp * comp_loss +
             lambda_boundary * boundary_value + lambda_router * router_total)
    return {
        "total": total, "global": global_loss, "fuse": fuse_loss, "comp": comp_loss,
        "boundary": boundary_value, "density": density_value, "route": route_value, "binary": binary_value,
    }


__all__ = ["compute_source_objective"]
