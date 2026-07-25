"""One explicit EMA adaptation optimization step."""

from __future__ import annotations

from loss.real.objective import compute_adaptation_objective

from .ema_core import update_teacher_after_success
from .step_control import perform_optimizer_step


def run_ema_adaptation_step(student, teacher, optimizer, real_loss_fn, anchor_loss_fn, *,
                            lambda_anchor, ema_decay, scheduler=None, max_grad_norm=None):
    """Execute the fixed EMA order and commit teacher state only on success.

    The supplied callbacks deliberately make real A/B/S and complete source
    anchor construction explicit at the entry point while keeping the commit
    order independently testable.
    """
    optimizer.zero_grad(set_to_none=True)
    real_loss = real_loss_fn()
    source_loss = anchor_loss_fn()
    objective = compute_adaptation_objective(real_loss, source_loss, lambda_anchor=lambda_anchor)
    objective["L_adapt"].backward()
    succeeded = perform_optimizer_step(optimizer, student.parameters(), max_grad_norm=max_grad_norm)
    if succeeded:
        if scheduler is not None:
            scheduler.step()
        update_teacher_after_success(teacher, student, ema_decay)
    return {
        "L_real": objective["L_real"].detach(),
        "L_src": objective["L_src"].detach(),
        "L_adapt": objective["L_adapt"].detach(),
        "step_succeeded": succeeded,
    }
