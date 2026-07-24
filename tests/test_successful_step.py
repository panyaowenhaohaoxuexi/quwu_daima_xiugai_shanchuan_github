import torch
from torch import nn

from training.step_control import perform_optimizer_step


def test_optimizer_step_reports_success_and_advances_parameter():
    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    before = model.weight.detach().clone()
    (model(torch.ones(1, 1)).sum()).backward()

    assert perform_optimizer_step(optimizer, model.parameters()) is True
    assert not torch.equal(before, model.weight)


def test_nonfinite_gradient_skips_optimizer_step():
    model = nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    before = model.weight.detach().clone()
    model.weight.grad = torch.full_like(model.weight, float("nan"))

    assert perform_optimizer_step(optimizer, model.parameters()) is False
    assert torch.equal(before, model.weight)
