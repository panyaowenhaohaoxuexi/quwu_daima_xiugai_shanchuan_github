"""V2 Source batch routing with physical completion-mask supervision."""

from loss.source import compute_physical_mask_losses


def linear_anneal(start, end, global_step, anneal_steps):
    """Linearly interpolate from ``start`` to ``end`` over non-negative steps."""
    if anneal_steps < 0:
        raise ValueError("anneal_steps must be non-negative")
    if anneal_steps == 0 or global_step >= anneal_steps:
        return float(end)
    if global_step <= 0:
        return float(start)
    progress = float(global_step) / anneal_steps
    return float(start) + (float(end) - float(start)) * progress


def source_route_schedule(global_step, *, tau_start, tau_end, temperature_anneal_steps,
                          teacher_anneal_steps):
    """Return the V2 temperature and GT-to-predicted-route teacher schedule."""
    return {
        "route_temperature": linear_anneal(tau_start, tau_end, global_step, temperature_anneal_steps),
        "teacher_gate_alpha": (0.0 if teacher_anneal_steps == 0 else
                                max(0.0, 1.0 - float(global_step) / teacher_anneal_steps)),
    }


def compute_physical_mask_batch_losses(model, source_batch, args, global_step, *, reconstruction_criteria):
    """The sole Source loss entry point; supports full and density-partial Source batches."""
    if len(source_batch) == 5:
        hazy, clear, tir, density_gt, completion_mask_gt = source_batch
        density_valid = None
    elif len(source_batch) == 6:
        hazy, clear, tir, density_gt, completion_mask_gt, density_valid = source_batch
    else:
        raise ValueError("Source batch must contain five tensors plus an optional density_valid flag")
    state = source_route_schedule(
        global_step, tau_start=args.route_tau_start, tau_end=args.route_tau_end,
        temperature_anneal_steps=args.route_temperature_anneal_steps,
        teacher_anneal_steps=args.route_teacher_anneal_steps,
    )
    context = model.encode_context(hazy, tir, route_temperature=state["route_temperature"])
    gate = state["teacher_gate_alpha"] * completion_mask_gt + (1.0 - state["teacher_gate_alpha"]) * context["route_soft"]
    output = model.decode_with_route(context, route_mode="soft", route_override_value=gate)
    global_ssim_criterion, global_contrast_criterion = reconstruction_criteria
    losses = compute_physical_mask_losses(
        output["pred_clear"], clear, output["density_map"], density_gt, output["route_logits"],
        completion_mask_gt, route_for_reconstruction=gate, boundary_map=output["boundary_map"],
        hazy_rgb=hazy, global_ssim_criterion=global_ssim_criterion,
        global_contrast_criterion=global_contrast_criterion, lambda_density=args.lambda_density,
        density_valid=density_valid,
        lambda_route=args.lambda_route, density_smooth_l1_beta=args.density_smooth_l1_beta,
        lambda_global=args.lambda_global, lambda_fuse=args.lambda_fuse, lambda_comp=args.lambda_comp,
        lambda_boundary=args.lambda_boundary, global_l1_weight=args.global_l1_weight,
        global_ssim_weight=args.global_ssim_weight, global_contrast_weight=args.global_contrast_weight,
        region_l1_weight=args.region_l1_weight, region_gradient_weight=args.region_gradient_weight,
        region_ssim_weight=args.region_ssim_weight, reconstruction_ssim_window=args.reconstruction_ssim_window,
        reconstruction_min_valid_support=args.reconstruction_min_valid_support,
    )
    return {"output": output, "losses": losses, "gate": gate, "state": state}
