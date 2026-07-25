"""Shared-context counterfactual forward controls for source route supervision."""

import torch

from loss.synthetic.counterfactual import compute_q


def detached_context(context):
    """Detach every tensor in a nested context without rerunning an encoder."""
    if torch.is_tensor(context):
        return context.detach()
    if isinstance(context, dict):
        return {key: detached_context(value) for key, value in context.items()}
    if isinstance(context, tuple):
        return tuple(detached_context(value) for value in context)
    if isinstance(context, list):
        return [detached_context(value) for value in context]
    return context


def gather_context(context, owner_indices):
    """Gather only a counterfactual chunk's owners from detached shared context."""
    if not isinstance(context, dict) or "density_map" not in context:
        raise ValueError("context must be the formal encode_context dictionary")
    owners = owner_indices.detach().long()
    batch_size = context["density_map"].shape[0]

    def _gather(value):
        if torch.is_tensor(value):
            if value.ndim > 0 and value.shape[0] == batch_size:
                return value.index_select(0, owners)
            return value
        if isinstance(value, dict):
            return {key: _gather(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(_gather(item) for item in value)
        if isinstance(value, list):
            return [_gather(item) for item in value]
        return value

    return _gather(context)


def build_counterfactual_routes(context, omega_support):
    """Build the two explicit route overrides over the original-size Omega support."""
    support = omega_support.detach().clamp(0, 1)
    return (
        {"route_override_value": torch.zeros_like(support), "route_override_mask": support,
         "memory_exclude_mask": torch.zeros_like(support)},
        {"route_override_value": torch.ones_like(support), "route_override_mask": support,
         "memory_exclude_mask": support},
    )


def run_counterfactual_pair(model, context, omega_support, route_mode="hard", boundary_mode="soft"):
    """Decode fusion/completion candidates without rerunning shared encoders."""
    context = detached_context(context)
    fusion_args, completion_args = build_counterfactual_routes(context, omega_support)
    with torch.no_grad():
        fusion = model.decode_with_route(context, route_mode=route_mode, boundary_mode=boundary_mode, **fusion_args)
        completion = model.decode_with_route(context, route_mode=route_mode, boundary_mode=boundary_mode, **completion_args)
    return fusion, completion


def run_counterfactual_chunks(model, context, owner_indices, omega_support,
                              chunk_size, route_mode="hard", boundary_mode="soft"):
    """Run candidate decoding in bounded owner-index chunks without encoding."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    owners, support = owner_indices.detach().long(), omega_support.detach()
    if owners.ndim != 1 or owners.numel() != support.shape[0]:
        raise ValueError("owner_indices must match the number of Omega supports")
    fusion_predictions, completion_predictions = [], []
    for start in range(0, owners.numel(), chunk_size):
        end = min(start + chunk_size, owners.numel())
        chunk_context = gather_context(detached_context(context), owners[start:end])
        fusion, completion = run_counterfactual_pair(
            model, chunk_context, support[start:end], route_mode=route_mode, boundary_mode=boundary_mode
        )
        fusion_predictions.append(fusion["pred_clear"])
        completion_predictions.append(completion["pred_clear"])
    if not fusion_predictions:
        empty = support.new_zeros((0, 3, *support.shape[-2:]))
        return {"pred_clear": empty}, {"pred_clear": empty}
    return ({"pred_clear": torch.cat(fusion_predictions, dim=0)},
            {"pred_clear": torch.cat(completion_predictions, dim=0)})


__all__ = [
    "detached_context", "gather_context", "build_counterfactual_routes",
    "run_counterfactual_pair", "run_counterfactual_chunks", "compute_q",
]
