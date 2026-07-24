"""Deterministic whole-batch paired 90-degree geometry for EMA."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GeometryTransform:
    rot90_k: int = 0
    horizontal_flip: bool = False

    def __post_init__(self):
        object.__setattr__(self, "rot90_k", int(self.rot90_k) % 4)

    def apply(self, tensor):
        output = torch.rot90(tensor, self.rot90_k, dims=(-2, -1))
        return torch.flip(output, dims=(-1,)) if self.horizontal_flip else output

    def inverse(self, tensor):
        output = torch.flip(tensor, dims=(-1,)) if self.horizontal_flip else tensor
        return torch.rot90(output, (-self.rot90_k) % 4, dims=(-2, -1))

    def state_dict(self):
        return {"rot90_k": self.rot90_k, "horizontal_flip": self.horizontal_flip}


def sample_geometry(generator):
    return GeometryTransform(
        rot90_k=int(torch.randint(0, 4, (), generator=generator)),
        horizontal_flip=bool(torch.randint(0, 2, (), generator=generator)),
    )
