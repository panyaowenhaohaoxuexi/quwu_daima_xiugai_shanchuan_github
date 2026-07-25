"""EMA parameter and buffer state updates with loss compatibility re-exports."""

import torch

from loss.real.consistency import real_consistency_loss, stability_weights, weighted_bce, weighted_l1


@torch.no_grad()
def ema_update(teacher, student, decay):
    if not 0 <= decay < 1:
        raise ValueError("decay must be in [0,1)")
    for teacher_parameter, student_parameter in zip(teacher.parameters(), student.parameters()):
        teacher_parameter.mul_(decay).add_(student_parameter, alpha=1.0 - decay)
        teacher_parameter.requires_grad_(False)


@torch.no_grad()
def synchronize_frozen_buffers(teacher, student):
    """Copy non-EMA buffers exactly, including frozen BN statistics/constants."""
    teacher_buffers = dict(teacher.named_buffers())
    student_buffers = dict(student.named_buffers())
    if teacher_buffers.keys() != student_buffers.keys():
        raise ValueError("teacher and student buffer structures differ")
    for name, teacher_buffer in teacher_buffers.items():
        if name == "ema_state" or name.endswith(".ema_state"):
            continue
        teacher_buffer.copy_(student_buffers[name])


@torch.no_grad()
def update_explicit_ema_state_buffers(teacher, student, decay):
    """EMA only buffers explicitly named ``ema_state``; never infer from dtype."""
    if not 0 <= decay < 1:
        raise ValueError("decay must be in [0,1)")
    teacher_buffers = dict(teacher.named_buffers())
    student_buffers = dict(student.named_buffers())
    for name, teacher_buffer in teacher_buffers.items():
        if name == "ema_state" or name.endswith(".ema_state"):
            source = student_buffers[name]
            if not torch.is_floating_point(teacher_buffer):
                raise ValueError("ema_state buffers must be floating point")
            teacher_buffer.mul_(decay).add_(source, alpha=1.0 - decay)


@torch.no_grad()
def update_teacher_after_success(teacher, student, decay):
    """The only teacher-update hook; caller invokes it after optimizer success."""
    ema_update(teacher, student, decay)
    synchronize_frozen_buffers(teacher, student)
    update_explicit_ema_state_buffers(teacher, student, decay)


__all__ = [
    "ema_update", "synchronize_frozen_buffers", "update_explicit_ema_state_buffers",
    "update_teacher_after_success", "stability_weights", "weighted_l1", "weighted_bce",
    "real_consistency_loss",
]
