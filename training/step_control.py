"""Optimizer-step helpers with explicit success semantics.

Training state (samplers, EMA teacher and counters) must only advance after a
student optimizer update actually succeeds.  This module intentionally keeps
that decision in one small, testable place.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch


def gradients_are_finite(parameters: Iterable[torch.nn.Parameter]) -> bool:
    """Return whether every materialized gradient contains finite values."""
    for parameter in parameters:
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            return False
    return True


def perform_optimizer_step(
    optimizer: torch.optim.Optimizer,
    parameters: Iterable[torch.nn.Parameter],
    *,
    scaler: torch.cuda.amp.GradScaler | None = None,
    max_grad_norm: float | None = None,
) -> bool:
    """Step ``optimizer`` and return ``True`` only when an update was applied.

    The caller owns ``zero_grad`` and all state commits.  On invalid gradients
    no optimizer state is touched.  For AMP, GradScaler signals an overflow by
    reducing its scale after ``update``; such skipped updates return ``False``.
    """
    params = [parameter for parameter in parameters if parameter.requires_grad]

    if scaler is not None:
        scaler.unscale_(optimizer)

    if not gradients_are_finite(params):
        return False

    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
        if not gradients_are_finite(params):
            return False

    if scaler is None:
        optimizer.step()
        return True

    scale_before = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    # GradScaler lowers its scale when an overflow makes ``step`` a no-op.
    return float(scaler.get_scale()) >= scale_before

