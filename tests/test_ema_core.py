import torch
from torch import nn

from EMA import real_consistency_loss, stability_weights, update_teacher_after_success


def test_stability_weights_are_separate_detached_and_decrease_with_teacher_difference():
    a = torch.zeros(1, 3, 2, 2, requires_grad=True)
    b = torch.ones(1, 3, 2, 2, requires_grad=True)
    weights = stability_weights(a, b, a[:, :1], b[:, :1], a[:, :1], b[:, :1], 1.0, 1.0, 1.0, 0.0)

    assert all(not value.requires_grad for value in weights)
    assert weights[0].mean() < 1.0
    losses = real_consistency_loss(a, b.detach(), a[:, :1], b[:, :1].detach(), a[:, :1], b[:, :1].detach(), *weights)
    assert set(losses) == {"L_J", "L_M", "L_R", "L_real"}


def test_ema_update_changes_parameters_only_after_a_successful_step():
    student = nn.Linear(1, 1, bias=False)
    teacher = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        student.weight.fill_(3.0)
        teacher.weight.fill_(1.0)

    update_teacher_after_success(teacher, student, decay=0.5)

    assert teacher.weight.item() == 2.0
    assert not teacher.weight.requires_grad


def test_ema_buffer_updates_copy_bn_and_constants_but_ema_only_explicit_state():
    class Buffered(nn.Module):
        def __init__(self):
            super().__init__()
            self.bn = nn.BatchNorm2d(1)
            self.register_buffer("cdc_kernel", torch.tensor([1.0]))
            self.register_buffer("counter", torch.tensor(1, dtype=torch.long))
            self.register_buffer("ema_state", torch.tensor([1.0]))

    student, teacher = Buffered(), Buffered()
    with torch.no_grad():
        student.bn.running_mean.fill_(4.0)
        student.cdc_kernel.fill_(7.0)
        student.counter.fill_(8)
        student.ema_state.fill_(5.0)
        teacher.ema_state.fill_(1.0)

    update_teacher_after_success(teacher, student, decay=0.5)

    assert teacher.bn.running_mean.item() == 4.0
    assert teacher.cdc_kernel.item() == 7.0
    assert teacher.counter.item() == 8
    assert teacher.ema_state.item() == 3.0
