"""Composable formal source-domain objective on the single final prediction."""

from loss.fog_routed_source_loss import (
    binary_route_penalty, masked_bce, masked_gradient_error, masked_local_ssim_error,
    masked_mean, masked_smooth_l1,
)


def _reconstruction_error(prediction, target, weight, *, l1_weight, gradient_weight,
                          ssim_weight, ssim_window, min_valid_support):
    l1 = masked_mean((prediction - target).abs(), weight)
    gradient = masked_gradient_error(prediction, target, weight)
    ssim = masked_local_ssim_error(prediction, target, weight, ssim_window, min_valid_support)
    return l1_weight * l1 + gradient_weight * gradient + ssim_weight * ssim


def compute_source_objective(pred_clear, clear_rgb, density_map, density_gt, route_soft,
                             boundary_map, q, omega_support, omega_weight=None,
                             density_beta=0.1, lambda_global=1.0, lambda_fuse=1.0,
                             lambda_comp=1.0, lambda_boundary=1.0, lambda_density=1.0,
                             lambda_route=1.0, lambda_binary=1.0, lambda_router=1.0,
                             rec_l1_weight=1.0, rec_gradient_weight=0.0, rec_ssim_weight=0.0,
                             boundary_l1_weight=1.0, boundary_gradient_weight=0.0,
                             ssim_window=7, min_valid_support=4):
    validity = density_gt.new_ones(density_gt.shape)
    r = route_soft.detach()
    b = boundary_map.detach()
    global_loss = _reconstruction_error(
        pred_clear, clear_rgb, validity, l1_weight=rec_l1_weight,
        gradient_weight=rec_gradient_weight, ssim_weight=rec_ssim_weight,
        ssim_window=ssim_window, min_valid_support=min_valid_support,
    )
    fuse_loss = _reconstruction_error(
        pred_clear, clear_rgb, (1 - r) * validity, l1_weight=rec_l1_weight,
        gradient_weight=rec_gradient_weight, ssim_weight=rec_ssim_weight,
        ssim_window=ssim_window, min_valid_support=min_valid_support,
    )
    comp_loss = _reconstruction_error(
        pred_clear, clear_rgb, r * validity, l1_weight=rec_l1_weight,
        gradient_weight=rec_gradient_weight, ssim_weight=rec_ssim_weight,
        ssim_window=ssim_window, min_valid_support=min_valid_support,
    )
    boundary_loss = (boundary_l1_weight * masked_mean((pred_clear - clear_rgb).abs(), b * validity) +
                     boundary_gradient_weight * masked_gradient_error(pred_clear, clear_rgb, b * validity))
    density_loss = masked_smooth_l1(density_map, density_gt, validity, density_beta)
    route_loss = masked_bce(route_soft, q.detach(), omega_support.detach(), omega_weight)
    binary_loss = binary_route_penalty(route_soft, validity)
    router_total = (lambda_density * density_loss + lambda_route * route_loss +
                    lambda_binary * binary_loss)
    total = (lambda_global * global_loss + lambda_fuse * fuse_loss + lambda_comp * comp_loss +
             lambda_boundary * boundary_loss + lambda_router * router_total)
    return {"total": total, "global": global_loss, "fuse": fuse_loss, "comp": comp_loss,
            "boundary": boundary_loss, "density": density_loss, "route": route_loss, "binary": binary_loss}
