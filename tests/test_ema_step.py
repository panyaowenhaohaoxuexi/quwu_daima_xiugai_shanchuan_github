import torch
from torch import nn

from training.ema_step import run_ema_adaptation_step


def test_ema_step_calls_real_and_anchor_once_and_updates_teacher_after_success():
    student = nn.Linear(1, 1, bias=False)
    teacher = nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        student.weight.fill_(2.0)
        teacher.weight.fill_(0.0)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    calls = {"real": 0, "anchor": 0}

    def real_loss():
        calls["real"] += 1
        return (student.weight - 1.0).square().sum()

    def anchor_loss():
        calls["anchor"] += 1
        return (student.weight - 3.0).square().sum()

    result = run_ema_adaptation_step(
        student, teacher, optimizer, real_loss, anchor_loss, lambda_anchor=0.5, ema_decay=0.5,
    )

    assert calls == {"real": 1, "anchor": 1}
    assert result["step_succeeded"] is True
    assert teacher.weight.item() != 0.0


def test_failed_ema_step_does_not_update_teacher_or_commit():
    student = nn.Linear(1, 1, bias=False)
    teacher = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    before = teacher.weight.detach().clone()

    result = run_ema_adaptation_step(
        student, teacher, optimizer,
        lambda: student.weight.sum() * torch.tensor(float("nan")),
        lambda: student.weight.sum() * 0.0,
        lambda_anchor=1.0, ema_decay=0.5,
    )

    assert result["step_succeeded"] is False
    assert torch.equal(teacher.weight, before)
