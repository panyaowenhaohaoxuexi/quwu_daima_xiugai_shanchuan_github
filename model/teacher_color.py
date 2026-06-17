import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalSemanticColorTransport(nn.Module):
    """
    Cross-modal semantic color transport.

    Query comes from full-image IR semantics. Key comes from reliable-region
    hazy-visible semantics. Value always comes from reliable-region x_vis_01
    RGB, never from clear_gt.
    """

    def __init__(self, in_channels=256, semantic_dim=128, num_prototypes=32, temperature=0.07, eps=1e-6):
        super(CrossModalSemanticColorTransport, self).__init__()
        self.semantic_dim = semantic_dim
        self.num_prototypes = num_prototypes
        self.temperature = temperature
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
        self.assignment_head = nn.Conv2d(semantic_dim, num_prototypes, kernel_size=1)

    def _safe_reliable_mean_rgb(self, rgb_value, reliable_mask, reliable_area):
        reliable_sum = reliable_mask.sum(dim=(2, 3), keepdim=False).clamp_min(self.eps)
        reliable_mean = (rgb_value * reliable_mask).sum(dim=(2, 3), keepdim=False) / reliable_sum
        safe_rgb = torch.full_like(reliable_mean, 0.5)
        has_reliable = (reliable_area > 0).view(-1, 1)
        return torch.where(has_reliable, reliable_mean, safe_rgb)

    def forward(self, ir_feat, vis_feat, x_vis_01, haze_mask):
        B, C, Hf, Wf = ir_feat.shape
        if vis_feat.shape[:2] != (B, C) or vis_feat.shape[2:] != (Hf, Wf):
            raise ValueError(f"IR/VIS feature mismatch: ir={ir_feat.shape}, vis={vis_feat.shape}")
        if haze_mask.dim() == 3:
            haze_mask = haze_mask.unsqueeze(1)

        s_ir = F.normalize(self.proj_ir(ir_feat), dim=1, eps=self.eps)
        s_vis = F.normalize(self.proj_vis(vis_feat), dim=1, eps=self.eps)

        mask_feat = F.interpolate(haze_mask.float(), size=(Hf, Wf), mode="nearest")
        mask_feat = (mask_feat >= 0.5).float()
        reliable_mask = 1.0 - mask_feat
        reliable_area_ratio = reliable_mask.mean(dim=(1, 2, 3))

        rgb_value = F.interpolate(x_vis_01.clamp(0.0, 1.0), size=(Hf, Wf), mode="bilinear", align_corners=False)
        rgb_value = rgb_value.clamp(0.0, 1.0)

        N = Hf * Wf
        assign_logits = self.assignment_head(s_vis).flatten(2)  # B,K,N
        reliable_flat = reliable_mask.flatten(2)  # B,1,N
        assign_logits = assign_logits + (1.0 - reliable_flat) * (-1e4)
        proto_assign = F.softmax(assign_logits, dim=-1)
        proto_assign = proto_assign * reliable_flat
        denom = proto_assign.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        proto_assign = proto_assign / denom
        has_reliable = (reliable_flat.sum(dim=-1, keepdim=True) > 0).to(proto_assign.dtype)
        proto_assign = proto_assign * has_reliable
        proto_assign = torch.nan_to_num(proto_assign, nan=0.0, posinf=0.0, neginf=0.0)

        s_vis_flat = s_vis.flatten(2).transpose(1, 2)  # B,N,D
        rgb_flat = rgb_value.flatten(2).transpose(1, 2)  # B,N,3
        proto_keys = torch.bmm(proto_assign, s_vis_flat)  # B,K,D
        proto_keys = F.normalize(proto_keys, dim=-1, eps=self.eps)
        proto_keys = torch.nan_to_num(proto_keys, nan=0.0, posinf=0.0, neginf=0.0)

        proto_values = torch.bmm(proto_assign, rgb_flat)  # B,K,3
        fallback_rgb = self._safe_reliable_mean_rgb(rgb_value, reliable_mask, reliable_area_ratio)
        proto_values = torch.where(has_reliable.transpose(1, 2).bool(), proto_values, fallback_rgb.unsqueeze(1))
        proto_values = torch.nan_to_num(proto_values, nan=0.5, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

        s_ir_flat = s_ir.flatten(2).transpose(1, 2)  # B,N,D
        logits = torch.bmm(s_ir_flat, proto_keys.transpose(1, 2)) / max(self.temperature, self.eps)
        proto_attn = F.softmax(logits, dim=-1)
        proto_attn = torch.nan_to_num(proto_attn, nan=1.0 / self.num_prototypes, posinf=0.0, neginf=0.0)
        proto_attn = proto_attn / proto_attn.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        transported_rgb_flat = torch.bmm(proto_attn, proto_values)  # B,N,3
        transported_rgb_feat = transported_rgb_flat.transpose(1, 2).reshape(B, 3, Hf, Wf)
        transported_rgb = F.interpolate(
            transported_rgb_feat,
            size=x_vis_01.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).clamp(0.0, 1.0)

        max_sim, _ = logits.mul(max(self.temperature, self.eps)).max(dim=-1)
        max_sim_map = max_sim.view(B, 1, Hf, Wf)
        finite_max_sim = torch.nan_to_num(max_sim, nan=0.0, posinf=0.0, neginf=0.0)
        max_sim_stats = {
            "mean": finite_max_sim.mean(dim=1),
            "max": finite_max_sim.max(dim=1).values,
            "min": finite_max_sim.min(dim=1).values,
        }

        return {
            "transported_rgb": transported_rgb,
            "transported_rgb_feat": transported_rgb_feat.clamp(0.0, 1.0),
            "semantic_ir": s_ir,
            "semantic_vis": s_vis,
            "proto_keys": proto_keys,
            "proto_values": proto_values,
            "proto_attn": proto_attn,
            "proto_assign": proto_assign.view(B, self.num_prototypes, Hf, Wf),
            "reliable_area_ratio": reliable_area_ratio,
            "max_sim_map": max_sim_map,
            "max_sim_stats": max_sim_stats,
        }
