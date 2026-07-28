"""Detached, deterministic local Omega sampler for counterfactual route labels."""

from collections import deque

import torch
from torch.nn import functional as F

from loss.coa_reconstruction import CoAContrastLoss, CoASSIM
from loss.source import compute_q, compute_source_objective


def build_source_reconstruction_criteria(device):
    """Create CoA's global SSIM and VGG-19 contrast losses once per process."""
    return CoASSIM().to(device), CoAContrastLoss().to(device)


class OmegaSampler:
    def __init__(self, regions_per_image=6, min_area=16, max_area=256, seed=0, edge_threshold=0.5):
        if not 4 <= regions_per_image <= 8:
            raise ValueError("regions_per_image must be in [4, 8]")
        if not 0 < min_area <= max_area:
            raise ValueError("invalid Omega area range")
        self.regions_per_image = int(regions_per_image)
        self.min_area = int(min_area)
        self.max_area = int(max_area)
        self.seed = int(seed)
        self.edge_threshold = float(edge_threshold)

    @staticmethod
    def _local_density_scale(image, y, x):
        """A detached seed-local density tolerance for connected growth."""
        radius = 2
        y0, y1 = max(0, y - radius), min(image.shape[0], y + radius + 1)
        x0, x1 = max(0, x - radius), min(image.shape[1], x + radius + 1)
        return image[y0:y1, x0:x1].std().clamp_min(0.02)

    def _grow_region(self, image, image_edge, accepted, y, x):
        """Grow one connected component without crossing an edge barrier."""
        height, width = image.shape
        scale = self._local_density_scale(image, y, x)
        seed_density = image[y, x]
        region = torch.zeros_like(accepted)
        visited = torch.zeros_like(accepted)
        queue = deque([(y, x)])
        visited[y, x] = True
        target_area = min(self.max_area, max(self.min_area, int(round(self.min_area * 1.5))))

        while queue and int(region.sum()) < target_area:
            cy, cx = queue.popleft()
            if accepted[cy, cx] or image_edge[cy, cx] > self.edge_threshold:
                continue
            if (image[cy, cx] - seed_density).abs() > (1.5 * scale + 0.01):
                continue
            region[cy, cx] = True
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < height and 0 <= nx < width and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((ny, nx))

        if int(region.sum()) < self.min_area:
            return None, scale
        return region, scale

    def sample(self, density_gt, tir_edge, generator=None):
        density = density_gt.detach()
        edge = tir_edge.detach()
        if density.ndim != 4 or density.shape[1] != 1 or edge.shape != density.shape:
            raise ValueError("density_gt and tir_edge must match [B,1,H,W]")
        local_generator = generator or torch.Generator(device=density.device)
        if generator is None:
            local_generator.manual_seed(self.seed)
        supports, weights, owners, classes, statuses = [], [], [], [], []
        for batch_index in range(density.shape[0]):
            image = density[batch_index, 0]
            image_edge = edge[batch_index, 0]
            density_flat = bool(image.std().detach() < 1e-6)
            tir_flat = bool(image_edge.std().detach() < 1e-6)
            quantiles = torch.quantile(image.flatten(), torch.tensor([1 / 3, 2 / 3], device=image.device))
            accepted = torch.zeros_like(image, dtype=torch.bool)
            local_count = 0
            requested_classes = ("low", "middle", "high")
            used_quantile_fallback = False
            for attempt in range(self.regions_per_image * 16):
                if local_count >= self.regions_per_image:
                    break
                klass = requested_classes[local_count % len(requested_classes)]
                if klass == "low":
                    candidates = torch.where(image <= quantiles[0])
                elif klass == "middle":
                    candidates = torch.where((image > quantiles[0]) & (image <= quantiles[1]))
                else:
                    candidates = torch.where(image > quantiles[1])
                if candidates[0].numel() == 0:
                    # A missing quantile never turns into a fake target: choose
                    # a remaining valid seed but retain the requested class for
                    # coverage diagnostics.
                    candidates = torch.where((~accepted) & (image_edge <= self.edge_threshold))
                    used_quantile_fallback = True
                if candidates[0].numel() == 0:
                    break
                choice = int(torch.randint(candidates[0].numel(), (), generator=local_generator, device=image.device))
                y, x = int(candidates[0][choice]), int(candidates[1][choice])
                region, scale = self._grow_region(image, image_edge, accepted, y, x)
                if region is None or (region & accepted).any():
                    continue
                accepted |= region
                support = region.float().unsqueeze(0)
                similarity = torch.exp(-0.5 * ((image - image[y, x]) / scale).square())
                edge_affinity = torch.exp(-image_edge / max(self.edge_threshold, 1e-6))
                weight = (support[0] * similarity * edge_affinity).unsqueeze(0)
                supports.append(support)
                weights.append(weight)
                owners.append(batch_index)
                classes.append(klass)
                local_count += 1
            statuses.append({
                "requested_count": self.regions_per_image,
                "actual_count": local_count,
                "fallback_reason": (
                    "flat_density_and_tir" if density_flat and tir_flat else
                    "flat_density" if density_flat else
                    "flat_tir" if tir_flat else
                    "empty_quantile_fallback" if used_quantile_fallback else
                    "insufficient_nonoverlapping_regions" if local_count < self.regions_per_image else None
                ),
            })
        height, width = density.shape[-2:]
        if supports:
            support = torch.stack(supports).detach()
            weight = torch.stack(weights).detach()
            owner = torch.tensor(owners, device=density.device, dtype=torch.long)
        else:
            support = density.new_zeros((0, 1, height, width)).detach()
            weight = density.new_zeros((0, 1, height, width)).detach()
            owner = torch.empty(0, device=density.device, dtype=torch.long)
        return {
            "omega_support": support,
            "omega_weight": weight,
            "owner_index": owner,
            "quantile_class": classes,
            "status": statuses,
        }


def _linear_warmup(step, start_step, warmup_steps, maximum):
    if step < start_step or maximum == 0:
        return 0.0
    if warmup_steps <= 0:
        return float(maximum)
    return float(maximum) * min(1.0, (step - start_step) / warmup_steps)


def source_route_schedule(global_step, *, tau_start, tau_end, hard_start_step,
                          counterfactual_start_step, route_loss_start_step,
                          route_loss_warmup_steps, binary_loss_start_step,
                          binary_loss_warmup_steps, base_lambda_route,
                          base_lambda_binary, temperature_anneal_steps=None):
    """Return the route state for one Source optimization step."""
    if tau_start <= 0 or tau_end <= 0:
        raise ValueError("route temperatures must be positive")
    if counterfactual_start_step > route_loss_start_step:
        raise ValueError("counterfactual_start_step must be <= route_loss_start_step")
    if global_step < 0:
        raise ValueError("global_step must be non-negative")
    anneal_steps = hard_start_step if temperature_anneal_steps is None else temperature_anneal_steps
    temperature = float(tau_end) if anneal_steps <= 0 else float(
        tau_start + min(1.0, global_step / anneal_steps) * (tau_end - tau_start)
    )
    return {
        "route_temperature": temperature,
        "route_mode": "hard" if global_step >= hard_start_step else "soft",
        "execute_counterfactual": global_step >= counterfactual_start_step,
        "lambda_route": _linear_warmup(global_step, route_loss_start_step, route_loss_warmup_steps, base_lambda_route),
        "lambda_binary": _linear_warmup(global_step, binary_loss_start_step, binary_loss_warmup_steps, base_lambda_binary),
    }


def _detach_context(context):
    if torch.is_tensor(context):
        return context.detach()
    if isinstance(context, dict):
        return {key: _detach_context(value) for key, value in context.items()}
    if isinstance(context, tuple):
        return tuple(_detach_context(value) for value in context)
    if isinstance(context, list):
        return [_detach_context(value) for value in context]
    return context


def _gather_context(context, owners):
    owners = owners.detach().long()
    batch_size = context["density_map"].shape[0]

    def gather(value):
        if torch.is_tensor(value):
            return value.index_select(0, owners) if value.ndim and value.shape[0] == batch_size else value
        if isinstance(value, dict):
            return {key: gather(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(gather(item) for item in value)
        if isinstance(value, list):
            return [gather(item) for item in value]
        return value

    return gather(context)


def _counterfactual_predictions(model, context, owners, supports, chunk_size, route_mode):
    if chunk_size < 1:
        raise ValueError("counterfactual_chunk_size must be >= 1")
    fusion_predictions, completion_predictions = [], []
    for start in range(0, owners.numel(), chunk_size):
        end = min(start + chunk_size, owners.numel())
        chunk_context = _gather_context(_detach_context(context), owners[start:end])
        support = supports[start:end].detach().clamp(0, 1)
        fusion_args = {"route_override_value": torch.zeros_like(support), "route_override_mask": support,
                       "memory_exclude_mask": torch.zeros_like(support)}
        completion_args = {"route_override_value": torch.ones_like(support), "route_override_mask": support,
                           "memory_exclude_mask": support}
        with torch.no_grad():
            fusion = model.decode_with_route(chunk_context, route_mode=route_mode, boundary_mode="soft", **fusion_args)
            completion = model.decode_with_route(chunk_context, route_mode=route_mode, boundary_mode="soft", **completion_args)
        fusion_predictions.append(fusion["pred_clear"])
        completion_predictions.append(completion["pred_clear"])
    if not fusion_predictions:
        empty = supports.new_zeros((0, 3, *supports.shape[-2:]))
        return empty, empty
    return torch.cat(fusion_predictions), torch.cat(completion_predictions)


def compute_source_batch_losses(model, source_batch, args, omega_sampler, global_step, *,
                                force_anchor_mode=False, omega_generator=None, reconstruction_criteria=None):
    """Compute the complete Source batch result without mutating optimizer state."""
    hazy, clear, tir, density = source_batch
    state = ({"route_temperature": float(args.route_tau_end), "route_mode": "hard",
              "execute_counterfactual": True, "lambda_route": float(args.lambda_route),
              "lambda_binary": float(args.lambda_binary)} if force_anchor_mode else source_route_schedule(
        global_step, tau_start=args.route_tau_start, tau_end=args.route_tau_end,
        hard_start_step=args.route_hard_start_step, counterfactual_start_step=args.counterfactual_start_step,
        route_loss_start_step=args.route_loss_start_step, route_loss_warmup_steps=args.route_loss_warmup_steps,
        binary_loss_start_step=args.binary_loss_start_step, binary_loss_warmup_steps=args.binary_loss_warmup_steps,
        base_lambda_route=args.lambda_route, base_lambda_binary=args.lambda_binary,
    ))
    context = model.encode_context(hazy, tir, route_temperature=state["route_temperature"])
    output = model.decode_with_route(context, route_mode=state["route_mode"], boundary_mode="soft")
    q_sum, q_valid_sum = torch.zeros_like(density), torch.zeros_like(density)
    omega_support, omega_weight = torch.zeros_like(density), torch.zeros_like(density)
    edge = F.interpolate(context["tir_structure_pyramid"]["h2"].detach().abs().mean(dim=1, keepdim=True),
                         size=density.shape[-2:], mode="bilinear", align_corners=False)
    omega = omega_sampler.sample(density, edge, generator=omega_generator)
    owners, supports = omega["owner_index"], omega["omega_support"]
    candidate_valid = torch.zeros(0, device=density.device, dtype=torch.bool)
    if state["execute_counterfactual"] and supports.numel():
        fusion, completion = _counterfactual_predictions(model, context, owners, supports,
                                                          args.counterfactual_chunk_size, state["route_mode"])
        candidate_q, candidate_valid_map = compute_q(
            clear.index_select(0, owners), fusion, completion, supports, args.q_temperature,
            window_size=getattr(args, "q_window_size", 1), min_valid_support=getattr(args, "q_min_valid_support", 1),
            l1_weight=getattr(args, "q_l1_weight", 1.0), gradient_weight=getattr(args, "q_gradient_weight", 0.0),
            ssim_weight=getattr(args, "q_ssim_weight", 0.0),
        )
        q_sum.index_add_(0, owners, candidate_q * candidate_valid_map)
        q_valid_sum.index_add_(0, owners, candidate_valid_map)
        omega_support.index_add_(0, owners, supports)
        omega_weight.index_add_(0, owners, omega["omega_weight"])
        candidate_valid = candidate_valid_map.flatten(1).gt(0).any(dim=1)
    q = q_sum / q_valid_sum.clamp_min(1.0)
    route_support = (omega_support * (q_valid_sum > 0).to(omega_support.dtype)).detach()
    route_supervision = {"sampled_omega_count": int(supports.shape[0]),
                          "valid_q_region_count": int(candidate_valid.sum().item()),
                          "valid_route_pixel_count": int((route_support > 0).sum().item())}
    global_ssim_criterion, global_contrast_criterion = (reconstruction_criteria or (None, None))
    losses = compute_source_objective(
        output["pred_clear"], clear, output["density_map"], density, output["route_soft"], output["boundary_map"],
        q, route_support, omega_weight, density_beta=args.density_smooth_l1_beta,
        lambda_global=args.lambda_global, lambda_fuse=args.lambda_fuse, lambda_comp=args.lambda_comp,
        lambda_boundary=args.lambda_boundary, lambda_router=args.lambda_router, lambda_density=args.lambda_density,
        lambda_route=state["lambda_route"], lambda_binary=state["lambda_binary"],
        hazy_rgb=hazy, global_ssim_criterion=global_ssim_criterion,
        global_contrast_criterion=global_contrast_criterion,
        global_l1_weight=getattr(args, "global_l1_weight", 0.8),
        global_ssim_weight=getattr(args, "global_ssim_weight", 0.2),
        global_contrast_weight=getattr(args, "global_contrast_weight", 0.0),
        region_l1_weight=getattr(args, "region_l1_weight", 1.0),
        region_gradient_weight=getattr(args, "region_gradient_weight", 0.2),
        region_ssim_weight=getattr(args, "region_ssim_weight", 0.2),
        ssim_window=args.reconstruction_ssim_window,
        min_valid_support=args.reconstruction_min_valid_support,
    )
    return {"context": context, "output": output, "losses": losses, "q": q,
            "q_valid_sum": q_valid_sum, "route_support": route_support,
            "route_supervision": route_supervision, "omega": omega, "state": state}
