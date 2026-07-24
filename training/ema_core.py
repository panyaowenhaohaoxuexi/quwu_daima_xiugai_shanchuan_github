"""Core EMA consistency math without data-loader or checkpoint side effects."""

import torch


def stability_weights(j_a, j_b, m_a, m_b, r_a, r_b, sigma_j, sigma_m, sigma_r, minimum):
    if min(sigma_j, sigma_m, sigma_r) <= 0:
        raise ValueError("EMA sigmas must be positive")
    w_j = torch.exp(-(j_a - j_b).abs().mean(dim=1, keepdim=True) / sigma_j)
    w_m = torch.exp(-(m_a - m_b).abs() / sigma_m)
    w_r = torch.exp(-(r_a - r_b).abs() / sigma_r)
    return tuple(weight.clamp(minimum, 1.0).detach() for weight in (w_j, w_m, w_r))


def weighted_l1(value, target, weight, eps=1e-6):
    expanded = weight.detach().expand_as(value)
    return (expanded * (value - target.detach()).abs()).sum() / expanded.sum().clamp_min(eps)


def weighted_bce(probability, target, weight, eps=1e-6):
    probability = probability.clamp(eps, 1.0 - eps)
    target, expanded = target.detach(), weight.detach().expand_as(probability)
    error = -(target * probability.log() + (1 - target) * (1 - probability).log())
    return (expanded * error).sum() / expanded.sum().clamp_min(eps)


def real_consistency_loss(j_student, j_target, m_student, m_target, r_student, r_target,
                          w_j, w_m, w_r, lambda_j=1.0, lambda_m=1.0, lambda_r=1.0):
    l_j = weighted_l1(j_student, j_target, w_j)
    l_m = weighted_l1(m_student, m_target, w_m)
    l_r = weighted_bce(r_student, r_target, w_r)
    return {"L_J": l_j, "L_M": l_m, "L_R": l_r, "L_real": lambda_j * l_j + lambda_m * l_m + lambda_r * l_r}


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
