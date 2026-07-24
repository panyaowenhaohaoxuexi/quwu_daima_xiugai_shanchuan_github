"""Pure source-route scheduling helpers."""

from __future__ import annotations


def _linear_warmup(step: int, start_step: int, warmup_steps: int, maximum: float) -> float:
    if step < start_step or maximum == 0:
        return 0.0
    if warmup_steps <= 0:
        return float(maximum)
    return float(maximum) * min(1.0, (step - start_step) / warmup_steps)


def source_route_schedule(
    global_step: int,
    *,
    tau_start: float,
    tau_end: float,
    hard_start_step: int,
    counterfactual_start_step: int,
    route_loss_start_step: int,
    route_loss_warmup_steps: int,
    binary_loss_start_step: int,
    binary_loss_warmup_steps: int,
    base_lambda_route: float,
    base_lambda_binary: float,
    temperature_anneal_steps: int | None = None,
) -> dict[str, float | str | bool]:
    """Return the explicit route state for a source-domain optimization step."""
    if tau_start <= 0 or tau_end <= 0:
        raise ValueError("route temperatures must be positive")
    if counterfactual_start_step > route_loss_start_step:
        raise ValueError("counterfactual_start_step must be <= route_loss_start_step")
    if global_step < 0:
        raise ValueError("global_step must be non-negative")

    anneal_steps = hard_start_step if temperature_anneal_steps is None else temperature_anneal_steps
    if anneal_steps <= 0:
        temperature = float(tau_end)
    else:
        ratio = min(1.0, global_step / anneal_steps)
        temperature = float(tau_start + ratio * (tau_end - tau_start))
    lambda_route = _linear_warmup(global_step, route_loss_start_step, route_loss_warmup_steps, base_lambda_route)
    lambda_binary = _linear_warmup(global_step, binary_loss_start_step, binary_loss_warmup_steps, base_lambda_binary)
    return {
        "route_temperature": temperature,
        "route_mode": "hard" if global_step >= hard_start_step else "soft",
        "execute_counterfactual": global_step >= counterfactual_start_step,
        "lambda_route": lambda_route,
        "lambda_binary": lambda_binary,
    }
