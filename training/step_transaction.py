"""Rollback helpers for training attempts that did not update parameters."""

from __future__ import annotations

import torch

from .checkpointing import capture_rng_state, restore_rng_state


def snapshot_module_buffers(module):
    """Capture every buffer without copying model parameters or optimizer state."""
    return {name: buffer.detach().clone() for name, buffer in module.named_buffers()}


@torch.no_grad()
def restore_module_buffers(module, snapshot):
    current = dict(module.named_buffers())
    if current.keys() != snapshot.keys():
        raise ValueError("module buffer structure changed during training step")
    for name, buffer in current.items():
        saved = snapshot[name]
        if buffer.shape != saved.shape:
            raise ValueError(
                f"buffer shape changed during training step: {name}: "
                f"current={tuple(buffer.shape)}, saved={tuple(saved.shape)}"
            )
        buffer.copy_(saved)


def snapshot_step_transaction(*, modules, omega_generator=None, geometry_generator=None,
                              dataloader_generators=None):
    """Snapshot rollback-safe state before a forward/backward attempt.

    GradScaler is intentionally excluded: its post-overflow reduced scale is
    required for the retry to make progress.
    """
    return {
        "buffers": [snapshot_module_buffers(module) for module in modules],
        "rng_state": capture_rng_state(
            omega_generator=omega_generator,
            geometry_generator=geometry_generator,
            dataloader_generators=dataloader_generators,
        ),
    }


def rollback_step_transaction(snapshot, *, modules, omega_generator=None, geometry_generator=None,
                              dataloader_generators=None):
    if len(modules) != len(snapshot["buffers"]):
        raise ValueError("transaction module count mismatch")
    for module, buffers in zip(modules, snapshot["buffers"]):
        restore_module_buffers(module, buffers)
    restore_rng_state(
        snapshot["rng_state"], omega_generator=omega_generator,
        geometry_generator=geometry_generator,
        dataloader_generators=dataloader_generators,
    )
