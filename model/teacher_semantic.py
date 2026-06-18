import torch.nn as nn
import torch.nn.functional as F


class SharedSemanticProjection(nn.Module):
    """Project IR and visible features into one normalized semantic space."""

    def __init__(self, in_channels=256, semantic_dim=128, eps=1e-6):
        super(SharedSemanticProjection, self).__init__()
        self.eps = eps
        self.proj_ir = nn.Sequential(
            nn.Conv2d(in_channels, semantic_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(semantic_dim, semantic_dim, kernel_size=1),
        )
        self.proj_vis = nn.Sequential(
            nn.Conv2d(in_channels, semantic_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(semantic_dim, semantic_dim, kernel_size=1),
        )

    def forward(self, ir_feat, vis_feat):
        s_ir = F.normalize(self.proj_ir(ir_feat), dim=1, eps=self.eps)
        s_vis = F.normalize(self.proj_vis(vis_feat), dim=1, eps=self.eps)
        return s_ir, s_vis
