"""Feature-guided soft/hard completion router."""

import torch
from torch import nn


class FeatureGuidedRouter(nn.Module):
    """Predict a route from HDE density, visible, infrared, and difference features."""

    def __init__(self, hidden_channels=32):
        super().__init__()
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be >= 1")
        self.head = nn.Sequential(
            nn.Conv2d(195, hidden_channels, 1), nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1), nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
        )

    def forward(self, density_map, routing_features, temperature=1.0):
        if density_map.ndim != 4 or density_map.shape[1] != 1:
            raise ValueError("density_map must have shape [B,1,H,W]")
        if float(temperature) <= 0:
            raise ValueError("temperature must be > 0")
        names = ("fm_vis", "fm_ir", "struct_diff_gap", "struct_diff_gmp")
        if set(routing_features) != set(names):
            raise ValueError("routing_features must contain the formal HDE routing feature keys")
        features = [density_map] + [routing_features[name] for name in names]
        if any(value.shape[0] != density_map.shape[0] or value.shape[-2:] != density_map.shape[-2:] for value in features):
            raise ValueError("routing features must align with density_map")
        logits = self.head(torch.cat(features, dim=1))
        soft = torch.sigmoid(logits / float(temperature))
        return {"route_logits": logits, "route_soft": soft, "route_hard": (soft >= 0.5).to(soft.dtype)}
