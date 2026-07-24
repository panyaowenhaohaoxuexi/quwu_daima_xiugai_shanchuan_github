"""Shared complete source-domain loss step for Teacher and EMA anchors."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from .schedules import source_route_schedule
from .source_counterfactual import compute_q, run_counterfactual_chunks
from .source_objective import compute_source_objective


def compute_source_batch_losses(model, source_batch, args, omega_sampler, global_step, *,
                                force_anchor_mode=False, omega_generator=None):
    """Encode once, decode once, run both detached candidate groups and form L_src."""
    hazy, clear, tir, density = source_batch
    if force_anchor_mode:
        state = {
            "route_temperature": float(args.route_tau_end), "route_mode": "hard",
            "execute_counterfactual": True, "lambda_route": float(args.lambda_route),
            "lambda_binary": float(args.lambda_binary),
        }
    else:
        state = source_route_schedule(
            global_step, tau_start=args.route_tau_start, tau_end=args.route_tau_end,
            hard_start_step=args.route_hard_start_step,
            counterfactual_start_step=args.counterfactual_start_step,
            route_loss_start_step=args.route_loss_start_step,
            route_loss_warmup_steps=args.route_loss_warmup_steps,
            binary_loss_start_step=args.binary_loss_start_step,
            binary_loss_warmup_steps=args.binary_loss_warmup_steps,
            base_lambda_route=args.lambda_route, base_lambda_binary=args.lambda_binary,
        )
    context = model.encode_context(hazy, tir, route_temperature=state["route_temperature"])
    output = model.decode_with_route(
        context, route_mode=state["route_mode"], boundary_mode="soft"
    )
    q_sum = torch.zeros_like(density)
    q_valid_sum = torch.zeros_like(density)
    omega_support, omega_weight = torch.zeros_like(density), torch.zeros_like(density)
    edge = context["tir_structure_pyramid"]["h2"].detach().abs().mean(dim=1, keepdim=True)
    edge = F.interpolate(edge, size=density.shape[-2:], mode="bilinear", align_corners=False)
    omega = omega_sampler.sample(density, edge, generator=omega_generator)
    owners, supports = omega["owner_index"], omega["omega_support"]
    if state["execute_counterfactual"] and supports.numel():
        fusion, completion = run_counterfactual_chunks(
            model, context, owners, supports, args.counterfactual_chunk_size, state["route_mode"]
        )
        candidate_q, candidate_valid = compute_q(
            clear.index_select(0, owners), fusion["pred_clear"], completion["pred_clear"],
            supports, args.q_temperature, window_size=getattr(args, "q_window_size", 1),
            min_valid_support=getattr(args, "q_min_valid_support", 1),
            l1_weight=getattr(args, "q_l1_weight", 1.0),
            gradient_weight=getattr(args, "q_gradient_weight", 0.0),
            ssim_weight=getattr(args, "q_ssim_weight", 0.0),
        )
        q_sum.index_add_(0, owners, candidate_q * candidate_valid)
        q_valid_sum.index_add_(0, owners, candidate_valid)
        omega_support.index_add_(0, owners, supports)
        omega_weight.index_add_(0, owners, omega["omega_weight"])
    q = q_sum / q_valid_sum.clamp_min(1.0)
    route_support = (omega_support * (q_valid_sum > 0).to(omega_support.dtype)).detach()
    candidate_has_valid_q = (
        candidate_valid.flatten(1).gt(0).any(dim=1)
        if state["execute_counterfactual"] and supports.numel()
        else torch.zeros(0, device=density.device, dtype=torch.bool)
    )
    route_supervision = {
        "sampled_omega_count": int(supports.shape[0]),
        "valid_q_region_count": int(candidate_has_valid_q.sum().item()),
        "valid_route_pixel_count": int((route_support > 0).sum().item()),
    }
    losses = compute_source_objective(
        output["pred_clear"], clear, output["density_map"], density,
        output["route_soft"], output["boundary_map"], q, route_support, omega_weight,
        density_beta=args.density_smooth_l1_beta, lambda_global=args.lambda_global,
        lambda_fuse=args.lambda_fuse, lambda_comp=args.lambda_comp,
        lambda_boundary=args.lambda_boundary, lambda_router=args.lambda_router,
        lambda_density=args.lambda_density, lambda_route=state["lambda_route"],
        lambda_binary=state["lambda_binary"], rec_l1_weight=args.rec_l1_weight,
        rec_gradient_weight=args.rec_gradient_weight, rec_ssim_weight=args.rec_ssim_weight,
        boundary_l1_weight=args.boundary_l1_weight,
        boundary_gradient_weight=args.boundary_gradient_weight,
        ssim_window=args.reconstruction_ssim_window,
        min_valid_support=args.reconstruction_min_valid_support,
    )
    return {"context": context, "output": output, "losses": losses, "q": q,
            "q_valid_sum": q_valid_sum, "route_support": route_support,
            "route_supervision": route_supervision, "omega": omega, "state": state}
