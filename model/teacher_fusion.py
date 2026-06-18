import torch
import torch.nn as nn
import torch.nn.functional as F


class BiDirectionalSemanticFusion(nn.Module):
    """H/4 semantic retrieval with reverse semantic verification."""

    def __init__(
        self,
        in_channels=256,
        semantic_dim=128,
        temperature=0.07,
        verify_threshold=0.2,
        verify_temperature=0.1,
        eps=1e-6,
        shared_proj=None,
    ):
        super(BiDirectionalSemanticFusion, self).__init__()
        if shared_proj is None:
            raise ValueError("shared_proj is required for BiDirectionalSemanticFusion")
        self.in_channels = in_channels
        self.semantic_dim = semantic_dim
        self.temperature = temperature
        self.verify_threshold = verify_threshold
        self.verify_temperature = verify_temperature
        self.eps = eps
        # The parent Teacher owns this module; avoid duplicate state_dict paths.
        object.__setattr__(self, "shared_proj", shared_proj)

    def forward(self, F_vis, F_ir, density_map, mask, fusion_head):
        if fusion_head is None:
            raise ValueError("fusion_head is required for BiDirectionalSemanticFusion")
        if F_vis.shape != F_ir.shape:
            raise ValueError(f"IR/VIS feature mismatch: vis={F_vis.shape}, ir={F_ir.shape}")
        if F_vis.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} feature channels, got {F_vis.shape[1]}"
            )
        if density_map is None:
            raise ValueError("density_map is required for BiDirectionalSemanticFusion")
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        s_ir, s_vis = self.shared_proj(F_ir, F_vis)
        B, C, Hf, Wf = F_vis.shape
        s_ir_flat = s_ir.flatten(2).transpose(1, 2)
        s_vis_flat = s_vis.flatten(2).transpose(1, 2)
        F_ir_flat = F_ir.flatten(2).transpose(1, 2)

        # Full attention is O(B*N*N). Large H/4 maps may OOM; use chunked
        # attention in a future high-resolution implementation when N is large.
        raw_v2i = torch.bmm(s_vis_flat, s_ir_flat.transpose(1, 2))
        score_v2i = raw_v2i / max(self.temperature, self.eps)
        density_h4 = F.interpolate(
            density_map,
            size=(Hf, Wf),
            mode="bilinear",
            align_corners=False,
        )
        conf = (1.0 - density_h4).flatten(2).transpose(1, 2).clamp(0.05, 1.0)
        score_v2i = score_v2i * conf
        attn_v2i = torch.softmax(score_v2i, dim=-1)
        retrieved_ir_flat = torch.bmm(attn_v2i, F_ir_flat)

        raw_i2v = torch.bmm(s_ir_flat, s_vis_flat.transpose(1, 2))
        max_sim_i2v = raw_i2v.max(dim=-1).values
        verify_gate = torch.sigmoid(
            (max_sim_i2v - self.verify_threshold)
            / max(self.verify_temperature, self.eps)
        )
        inject_flat = verify_gate.unsqueeze(-1) * retrieved_ir_flat
        inject = inject_flat.transpose(1, 2).reshape(B, C, Hf, Wf)

        g_input = torch.cat([F_vis, inject, density_h4], dim=1)
        if g_input.shape[1] != 2 * self.in_channels + 1:
            raise ValueError(
                f"fusion head input must have {2 * self.in_channels + 1} channels, "
                f"got {g_input.shape[1]}"
            )
        g = fusion_head(g_input)
        mask = F.interpolate(mask.float(), size=(Hf, Wf), mode="nearest").to(F_vis.dtype)
        F_fused = (1.0 - mask) * (F_vis + g * inject) + mask * F_ir

        debug = {
            "attn_v2i": attn_v2i,
            "verify_gate": verify_gate.view(B, 1, Hf, Wf),
            "g": g,
            "inject": inject,
            "fusion_score_map": raw_v2i.max(dim=-1).values.view(B, 1, Hf, Wf),
            "verify_score_map": max_sim_i2v.view(B, 1, Hf, Wf),
        }
        return F_fused, debug
