"""CoA-equivalent fixed-step training utilities."""

import math

import torch


def cosine_decay_lr(step, total_steps, start_lr, end_lr):
    """Return CoA's cosine-decayed learning rate for a zero-based-or-later step."""
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if not 0 <= step <= total_steps:
        raise ValueError("step must be in [0, total_steps]")
    if start_lr <= 0 or end_lr <= 0:
        raise ValueError("learning rates must be positive")
    return end_lr + 0.5 * (start_lr - end_lr) * (1 + math.cos(step * math.pi / total_steps))


def cycle_batches(loader):
    """Yield batches indefinitely, restarting a non-empty iterable on exhaustion."""
    while True:
        yielded = False
        for batch in loader:
            yielded = True
            yield batch
        if not yielded:
            raise ValueError("cannot cycle an empty loader")


def build_coa_adam(parameters, *, learning_rate):
    """Build CoA's Adam optimizer without AdamW weight decay."""
    return torch.optim.Adam(parameters, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)


def set_cosine_learning_rate(optimizer, *, step, total_steps, start_lr, end_lr, no_lr_sche):
    """Set and return the CoA schedule value for one optimizer step."""
    learning_rate = start_lr if no_lr_sche else cosine_decay_lr(step, total_steps, start_lr, end_lr)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    return learning_rate
