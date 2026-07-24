import random

import numpy as np
import torch
from torch import nn

from training.step_transaction import rollback_step_transaction, snapshot_step_transaction


class _BufferedModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.batchnorm = nn.BatchNorm1d(2)
        self.register_buffer("ema_state", torch.tensor([3.0]))

    def forward(self, value):
        return self.batchnorm(value)


def test_rollback_restores_all_buffers_and_rng_generators_without_touching_parameters():
    random.seed(3)
    np.random.seed(3)
    torch.manual_seed(3)
    module = _BufferedModule().train()
    parameter_before = module.batchnorm.weight.detach().clone()
    buffers_before = {name: value.detach().clone() for name, value in module.named_buffers()}
    omega = torch.Generator().manual_seed(4)
    geometry = torch.Generator().manual_seed(5)
    source_loader = torch.Generator().manual_seed(6)
    transaction = snapshot_step_transaction(
        modules=[module], omega_generator=omega, geometry_generator=geometry,
        dataloader_generators={"source": source_loader},
    )
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())),
                float(torch.rand((), generator=omega)), float(torch.rand((), generator=geometry)),
                float(torch.rand((), generator=source_loader)))
    module(torch.rand(4, 2))
    module.ema_state.add_(9)
    module.batchnorm.weight.data.add_(2)
    random.random(); np.random.rand(); torch.rand(())
    torch.rand((), generator=omega); torch.rand((), generator=geometry); torch.rand((), generator=source_loader)

    rollback_step_transaction(
        transaction, modules=[module], omega_generator=omega, geometry_generator=geometry,
        dataloader_generators={"source": source_loader},
    )

    assert all(torch.equal(value, buffers_before[name]) for name, value in module.named_buffers())
    assert not torch.equal(module.batchnorm.weight, parameter_before)
    actual = (random.random(), float(np.random.rand()), float(torch.rand(())),
              float(torch.rand((), generator=omega)), float(torch.rand((), generator=geometry)),
              float(torch.rand((), generator=source_loader)))
    assert actual == expected
