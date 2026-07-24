"""Strictly pointwise monotonic fog-density router."""

import torch
from torch import nn
from torch.nn import functional as F


class MonotonicFogRouter(nn.Module):
    """Apply one learned monotonic scalar function independently to every pixel."""

    def __init__(self, hidden_channels=8, initial_bias=-4.0):
        super().__init__()
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be >= 1")
        self.raw_weight_in = nn.Parameter(torch.full((hidden_channels,), -2.0))
        self.bias_in = nn.Parameter(torch.zeros(hidden_channels))
        self.raw_weight_out = nn.Parameter(torch.full((hidden_channels,), -2.0))
        self.bias_out = nn.Parameter(torch.tensor(float(initial_bias)))

    def forward(self, density_map, temperature=1.0):
        if density_map.ndim != 4 or density_map.shape[1] != 1:
            raise ValueError("density_map must have shape [B, 1, H, W]")
        if float(temperature) <= 0:
            raise ValueError("temperature must be > 0")
        values = density_map[:, 0].unsqueeze(-1)
        hidden = F.relu(values * F.softplus(self.raw_weight_in) + self.bias_in)
        logits = (hidden * F.softplus(self.raw_weight_out)).sum(dim=-1, keepdim=False)
        route_logits = (logits + self.bias_out).unsqueeze(1)
        route_soft = torch.sigmoid(route_logits / float(temperature))
        route_binary = (route_soft >= 0.5).to(route_soft.dtype)
        route_hard = route_binary.detach() - route_soft.detach() + route_soft
        return {
            "route_logits": route_logits,
            "route_soft": route_soft,
            "route_hard": route_hard,
        }
