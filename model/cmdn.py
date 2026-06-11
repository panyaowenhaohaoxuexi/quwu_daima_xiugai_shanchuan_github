"""
model/cmdn.py — Cross-Modal Decision Network

Replaces HDE+differentiable_otsu for haze mask estimation.
Stable pseudo-label: haze_app × (DINOv2 structure degradation + fixed VIS-IR
structure advantage), where:
  haze_app   : fixed visible fog-white appearance prior
  attn_deg   : DINOv2 local structure variance (high=clear, low=degraded)
  struct_deg : 1 - attn_deg
  ir_adv     : fixed Sobel/local-contrast IR advantage over visible

g_fog is kept only as a CLIP sliding-window diagnostic signal. It does not
enter P_pseudo, P_support, or loss_Disc. CLIP patch tokens still enter the
decoder as feat_clip for trainable P_fail prediction.

Probe-verified design decisions (see probe_hooks.py):
  - CLIP per-patch tokens CANNOT encode fog/sky semantics (all cos-sim ~0).
    g_fog uses sliding-window CLS token (7×7 overlapping crops at 448px).
  - DINOv2 constructed with img_size=518, block_chunks=0, init_values=1.0,
    FlatBlock (strict match, 175/175 keys). 448 input via pos_embed interpolation.
  - attn_deg uses x_prenorm with 3×3 local structure variance
    (not x_norm_patchtokens nor global L2 norm — 5× better margin in probe).
  - CLIP feat_clip (decoder input) still uses per-patch tokens via hook.

Current region decision:
  P_fail -> tau -> G_dec, then conservative support P_support gates G_soft.
  The forward value uses binary M_hard, while gradients flow through G_soft.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def per_image_minmax(x):
    """Per-image min-max normalisation to [0,1]."""
    B = x.shape[0]
    x_flat = x.view(B, -1)
    x_min = x_flat.min(dim=1, keepdim=True)[0]
    x_max = x_flat.max(dim=1, keepdim=True)[0]
    tail = [1] * (x.dim() - 1)
    return (x - x_min.view(B, *tail)) / (x_max.view(B, *tail) - x_min.view(B, *tail) + 1e-8)


def _conv3x3_bn_relu(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


# ---------------------------------------------------------------------------
# CMDN
# ---------------------------------------------------------------------------

class CMDN(nn.Module):
    """Cross-Modal Decision Network for haze mask estimation.

    Args:
        dino_source_dir : path to DINOv2 source
        dino_weight_path: path to dinov2_vitb14_pretrain.pth
        gamma           : pseudo-label sparsification exponent (default 1.2)
    """

    def __init__(self,
                 dino_source_dir="./DINOv2/facebookresearch_dinov2_main",
                 dino_weight_path="./dinov2_model/dinov2_vitb14_pretrain.pth",
                 gamma=1.2,
                 tau_min=0.25,
                 tau_max=0.85,
                 gate_temperature=0.10,
                 support_gamma=1.0,
                 hard_gate_threshold=0.5):
        super().__init__()
        self.gamma = gamma
        self.tau_min = tau_min
        self.tau_max = tau_max
        self.gate_temperature = gate_temperature
        self.support_gamma = support_gamma
        self.hard_gate_threshold = hard_gate_threshold

        # ------------------------------------------------------------------
        # 1. Trainable VIS / IR encoders (lightweight, no weight sharing)
        # ------------------------------------------------------------------
        self.vis_enc = nn.Sequential(
            _conv3x3_bn_relu(3, 16),
            _conv3x3_bn_relu(16, 32),
            _conv3x3_bn_relu(32, 32),
        )
        self.ir_enc = nn.Sequential(
            _conv3x3_bn_relu(3, 16),
            _conv3x3_bn_relu(16, 32),
            _conv3x3_bn_relu(32, 32),
        )

        # ------------------------------------------------------------------
        # 2. Normalisation constants (register_buffer for .cuda() safety)
        # ------------------------------------------------------------------
        self.register_buffer(
            'clip_mean',
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer(
            'clip_std',
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
        self.register_buffer(
            'imagenet_mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer(
            'imagenet_std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer(
            'sobel_x',
            torch.tensor([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer(
            'sobel_y',
            torch.tensor([[-1, -2, -1],
                          [0, 0, 0],
                          [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))

        # ------------------------------------------------------------------
        # 3. CLIP loading (frozen)
        #    - g_fog:      sliding-window CLS token (image-level, 7x7 grid)
        #    - feat_clip:  per-patch tokens via hook (decoder input only)
        # ------------------------------------------------------------------
        try:
            import CLIP.clip as clip
        except Exception as e:
            raise RuntimeError(
                f"[CMDN] Cannot import CLIP.clip: {e}.")

        self._clip_model, _ = clip.load(
            "ViT-B/32", device=torch.device("cpu"),
            download_root="./clip_model/")
        for p in self._clip_model.parameters():
            p.requires_grad = False
        self._clip_model.eval()

        # Hook for per-patch tokens (feat_clip only)
        self.clip_visual = self._clip_model.visual
        self._clip_patch_tokens = None
        self.clip_visual.transformer.resblocks[-1].register_forward_hook(
            self._clip_hook)

        # Text anchors (fp32 for downstream matmul)
        with torch.no_grad():
            device_for_text = next(self._clip_model.parameters()).device
            t_fog_raw = self._clip_model.encode_text(clip.tokenize([
                "dense smoke",
                "thick fog obscuring objects",
                "smoke blocking the scene",
            ]).to(device_for_text)).mean(0)  # (512,)
            t_sky_raw = self._clip_model.encode_text(clip.tokenize([
                "clear sky",
                "overcast sky",
                "cloudy sky",
            ]).to(device_for_text)).mean(0)  # (512,)
        self.register_buffer('t_fog', t_fog_raw.float())
        self.register_buffer('t_sky', t_sky_raw.float())

        # ------------------------------------------------------------------
        # 4. DINOv2 loading (frozen, strict match)
        # ------------------------------------------------------------------
        if not os.path.isdir(dino_source_dir):
            raise RuntimeError(
                f"[CMDN] DINOv2 source dir not found: {dino_source_dir}")
        if not os.path.isfile(dino_weight_path):
            raise RuntimeError(
                f"[CMDN] DINOv2 weights not found: {dino_weight_path}")

        sys.path.insert(0, dino_source_dir)
        try:
            from dinov2.models.vision_transformer import DinoVisionTransformer
            from dinov2.layers.block import Block as FlatBlock
            from dinov2.layers import MemEffAttention
            from functools import partial
        except ImportError as e:
            sys.path.pop(0)
            raise RuntimeError(
                f"[CMDN] Cannot import DINOv2 modules: {e}.")

        self._dino = DinoVisionTransformer(
            img_size=518, patch_size=14, embed_dim=768, depth=12,
            num_heads=12, mlp_ratio=4,
            block_fn=partial(FlatBlock, attn_class=MemEffAttention),
            block_chunks=0, init_values=1.0,
        )

        state_dict = torch.load(dino_weight_path, map_location='cpu')
        try:
            self._dino.load_state_dict(state_dict, strict=True)
        except RuntimeError as e:
            sys.path.pop(0)
            raise RuntimeError(
                f"[CMDN] DINOv2 strict load failed: {e}")
        sys.path.pop(0)

        for p in self._dino.parameters():
            p.requires_grad = False
        self._dino.eval()

        # ------------------------------------------------------------------
        # 5. Projection layers (near-zero init)
        # ------------------------------------------------------------------
        self.clip_proj = nn.Conv2d(768, 32, 1, bias=False)
        nn.init.normal_(self.clip_proj.weight, std=1e-4)

        self.dino_proj = nn.Conv2d(768, 32, 1, bias=False)
        nn.init.normal_(self.dino_proj.weight, std=1e-4)

        # ------------------------------------------------------------------
        # 6. Decoder: 32(vis)+32(clip)+32(dino)+1(P_pseudo) = 97 → 1
        # ------------------------------------------------------------------
        self.decoder = nn.Sequential(
            nn.Conv2d(97, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid(),
        )

        self.thr_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(97, 32, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.thr_head[-2].weight)
        nn.init.zeros_(self.thr_head[-2].bias)

        # ------------------------------------------------------------------
        # 7. Sliding-window g_fog: pre-compute fixed grid positions
        #    Grid 7×7, each crop 224×224 resized from a 448×448 parent.
        #    Crop centre spacing = 448/7 = 64 px.
        # ------------------------------------------------------------------
        self._gfog_grid = 7
        self._gfog_step = 448.0 / 7.0  # 64.0
        self._gfog_half = 112           # half of 224

    # ------------------------------------------------------------------
    # CLIP hook callback (for per-patch feat_clip, not g_fog)
    # ------------------------------------------------------------------
    def _clip_hook(self, module, inp, out):
        self._clip_patch_tokens = out  # (seq, batch, dim) = (50, B, 768)

    # ------------------------------------------------------------------
    # Fixed image operators for stable pseudo-label generation
    # ------------------------------------------------------------------
    def _rgb_to_gray(self, x):
        return x[:, 0:1] * 0.299 + x[:, 1:2] * 0.587 + x[:, 2:3] * 0.114

    def _sobel_edge(self, gray, eps=1e-8):
        sobel_x = self.sobel_x.to(dtype=gray.dtype)
        sobel_y = self.sobel_y.to(dtype=gray.dtype)
        gx = F.conv2d(gray, sobel_x, padding=1)
        gy = F.conv2d(gray, sobel_y, padding=1)
        return torch.sqrt(gx * gx + gy * gy + eps)

    def _local_std(self, gray, kernel_size=7, eps=1e-8):
        pad = kernel_size // 2
        mean = F.avg_pool2d(gray, kernel_size, stride=1, padding=pad)
        mean_sq = F.avg_pool2d(gray * gray, kernel_size, stride=1, padding=pad)
        var = (mean_sq - mean * mean).clamp_min(0.0)
        return torch.sqrt(var + eps)

    def _robust_norm(self, x, low_q=0.02, high_q=0.98, min_range=1e-4):
        B = x.shape[0]
        flat = x.reshape(B, -1)

        try:
            low = torch.quantile(flat, low_q, dim=1, keepdim=True)
            high = torch.quantile(flat, high_q, dim=1, keepdim=True)
        except Exception:
            n = flat.shape[1]
            low_k = max(1, min(n, int((n - 1) * low_q) + 1))
            high_k = max(1, min(n, int((n - 1) * high_q) + 1))
            low = flat.kthvalue(low_k, dim=1, keepdim=True).values
            high = flat.kthvalue(high_k, dim=1, keepdim=True).values

        tail = [1] * (x.dim() - 1)
        low = low.view(B, *tail)
        high = high.view(B, *tail)
        span = high - low
        valid = span >= min_range
        norm = ((x - low) / (span + 1e-8)).clamp(0.0, 1.0)
        return torch.where(valid, norm, torch.zeros_like(norm))

    def _smooth01(self, x, kernel_size=5):
        pad = kernel_size // 2
        x = F.avg_pool2d(x, kernel_size, stride=1, padding=pad)
        return x.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Sliding-window g_fog via CLS token
    # ------------------------------------------------------------------
    def _compute_gfog_sliding(self, x_vis_01):
        """
        Args:
            x_vis_01: (B, 3, 448, 448) image in [0, 1], already resized.
        Returns:
            g_fog: (B, 1, H, W) fog probability map (high = foggy).
        """
        B, _, H, W = x_vis_01.shape
        clip_dtype = self.clip_visual.conv1.weight.dtype
        DEV = x_vis_01.device

        # Build grid centre coordinates
        step = self._gfog_step
        half = self._gfog_half
        G = self._gfog_grid

        cy_list = [int(step * i + step / 2) for i in range(G)]  # 32, 96, 160, ...
        cx_list = [int(step * j + step / 2) for j in range(G)]

        # Collect all crops: (B * G * G, 3, 224, 224)
        crops = []
        for cy in cy_list:
            for cx in cx_list:
                y1 = max(0, cy - half)
                x1 = max(0, cx - half)
                y2 = min(H, y1 + 224)
                x2 = min(W, x1 + 224)
                y1 = y2 - 224
                x1 = x2 - 224
                # Crop in [0,1] → CLIP normalise
                crop = x_vis_01[:, :, y1:y2, x1:x2]  # (B, 3, 224, 224)
                crop_norm = (crop - self.clip_mean) / self.clip_std
                crops.append(crop_norm)

        # Stack: (B*49, 3, 224, 224)
        crops_batch = torch.cat(crops, dim=0)

        # Single batched CLIP forward (much faster than 49 sequential calls)
        with torch.no_grad():
            cls_feats = self.clip_visual(crops_batch.type(clip_dtype))  # (B*49, 512)
        cls_feats = cls_feats.float()
        cls_feats = F.normalize(cls_feats, dim=-1)  # (B*49, 512)

        # Reshape: (B*49, 512) → (B, 49, 512)
        cls_feats = cls_feats.view(B, G * G, 512)

        # Cosine similarity with text anchors
        t_fog_n = F.normalize(self.t_fog, dim=0)  # (512,)
        t_sky_n = F.normalize(self.t_sky, dim=0)  # (512,)

        sim_fog = cls_feats @ t_fog_n  # (B, 49)
        sim_sky = cls_feats @ t_sky_n  # (B, 49)

        g_fog_raw = (sim_fog - sim_sky).reshape(B, 1, G, G)  # (B, 1, 7, 7)

        # Upsample to H×W, then per-image minmax
        g_fog = F.interpolate(g_fog_raw, size=(H, W),
                              mode='bilinear', align_corners=False)
        g_fog = per_image_minmax(g_fog)  # [0, 1]
        return g_fog

    # ------------------------------------------------------------------
    # DINOv2 local structure variance (3×3 neighbourhood)
    # ------------------------------------------------------------------
    def _compute_attn_deg(self, x_vis_imagenet_448, target_hw=None):
        """Returns feat_dino, attn_deg at target_hw (or 448 if None)."""
        B, _, H448, W448 = x_vis_imagenet_448.shape
        out_h, out_w = target_hw if target_hw else (H448, W448)

        with torch.no_grad():
            out = self._dino.forward_features(x_vis_imagenet_448)

        # x_prenorm (probe-verified: 32% better spatial discrimination than
        # x_norm_patchtokens).  Drop CLS token at index 0.
        pt_pre = out["x_prenorm"]         # (B, grid^2+1, 768)
        pt_pre_p = pt_pre[:, 1:, :]       # (B, grid^2, 768)

        grid = int(pt_pre_p.shape[1] ** 0.5)
        D = pt_pre_p.shape[-1]

        # Feature map for decoder input
        feat_dino_raw = pt_pre_p.reshape(B, grid, grid, D).permute(0, 3, 1, 2)  # (B,768,grid,grid)
        feat_dino = self.dino_proj(feat_dino_raw.float())
        feat_dino = F.interpolate(feat_dino, size=(out_h, out_w),
                                  mode='bilinear', align_corners=False)

        # Local structure variance (3×3)
        # Probe-verified: this gives 5× better smoke/clear margin than L2 norm.
        feat_3d = pt_pre_p.float().reshape(B, grid, grid, D)  # (B,grid,grid,768)
        feat_padded = F.pad(
            feat_3d.permute(0, 3, 1, 2),   # (B,768,grid,grid)
            (1, 1, 1, 1), mode='reflect')

        local_var = feat_3d.new_zeros(B, grid, grid)
        n_neighbours = 0
        for di in range(3):
            for dj in range(3):
                if di == 1 and dj == 1:
                    continue
                neighbour = feat_padded[:, :, di:di+grid, dj:dj+grid]
                diff = (feat_3d.permute(0, 3, 1, 2) - neighbour).norm(dim=1)
                local_var += diff
                n_neighbours += 1
        local_var = local_var / n_neighbours

        attn_deg = local_var.unsqueeze(1)       # (B, 1, grid, grid)
        attn_deg = per_image_minmax(attn_deg)   # [0, 1], high = clear
        attn_deg = F.interpolate(attn_deg, size=(out_h, out_w),
                                 mode='bilinear', align_corners=False)

        return feat_dino, attn_deg

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, x_vis_clipnorm, x_ir,
                disc_alpha=0.0, return_debug=False):
        """
        Args:
            x_vis_clipnorm: (B, 3, H, W) visible in CLIP normalisation
            x_ir:           (B, 3, H, W) infrared
            disc_alpha:     kept for interface compatibility; does not affect P_pseudo
            return_debug:   if True, also return pseudo-label diagnostics

        Returns:
            P_fail:    (B, 1, H, W) visible failure probability
            P_pseudo:  (B, 1, H, W) fixed pseudo-label (detached)
        """
        B, _, H, W = x_vis_clipnorm.shape

        # ================================================================
        # (a) Trainable visible feature for P_fail prediction
        # ================================================================
        f_vis = self.vis_enc(x_vis_clipnorm)

        # ================================================================
        # (b) Denormalise inputs for fixed image operators and CLIP
        # ================================================================
        x_vis_01 = (x_vis_clipnorm * self.clip_std + self.clip_mean).clamp(0, 1)
        x_ir_01 = (x_ir * self.clip_std + self.clip_mean).clamp(0, 1)

        # ================================================================
        # (c) CLIP features + sliding-window g_fog
        # ================================================================
        # --- g_fog: sliding-window CLS (image-level semantics) ---
        # Diagnostic only: not used in P_pseudo, P_support, or loss_Disc.
        x_vis_448 = F.interpolate(x_vis_01, size=(448, 448),
                                  mode='bilinear', align_corners=False)
        g_fog = self._compute_gfog_sliding(x_vis_448)  # (B, 1, 448, 448)
        g_fog = F.interpolate(g_fog, size=(H, W),
                              mode='bilinear', align_corners=False)  # → H×W

        # --- feat_clip: per-patch tokens via hook (decoder input only) ---
        x_vis_224 = F.interpolate(x_vis_01, size=(224, 224),
                                  mode='bilinear', align_corners=False)
        clip_dtype = self.clip_visual.conv1.weight.dtype
        _ = self.clip_visual(x_vis_224.type(clip_dtype))  # trigger hook

        raw = self._clip_patch_tokens             # (50, B, 768)
        patch_tokens = raw.permute(1, 0, 2)       # (B, 50, 768)
        patch_tokens = patch_tokens[:, 1:, :]      # (B, 49, 768)
        patch_tokens = patch_tokens.float()

        feat_clip = patch_tokens.reshape(B, 7, 7, 768).permute(0, 3, 1, 2)  # (B,768,7,7)
        feat_clip = self.clip_proj(feat_clip)                                # (B,32,7,7)
        feat_clip = F.interpolate(feat_clip, size=(H, W),
                                  mode='bilinear', align_corners=False)      # (B,32,H,W)

        # ================================================================
        # (d) DINOv2 degradation
        # ================================================================
        x_vis_imagenet = (x_vis_01 - self.imagenet_mean) / self.imagenet_std
        x_vis_dino = F.interpolate(x_vis_imagenet, size=(448, 448),
                                   mode='bilinear', align_corners=False)
        feat_dino, attn_deg = self._compute_attn_deg(x_vis_dino, target_hw=(H, W))

        # ================================================================
        # (e) Stable fixed pseudo-label generation
        # ================================================================
        # P_pseudo is fixed for a given VIS/IR input and does not depend on
        # trainable encoders, disc_alpha, or epoch schedules.
        with torch.no_grad():
            gray_vis = self._rgb_to_gray(x_vis_01)
            gray_ir = self._rgb_to_gray(x_ir_01)

            brightness = gray_vis
            saturation = x_vis_01.max(dim=1, keepdim=True).values - \
                x_vis_01.min(dim=1, keepdim=True).values
            local_contrast_vis = self._local_std(gray_vis)
            local_contrast_ir = self._local_std(gray_ir)

            bright_gate = torch.sigmoid((brightness - 0.60) / 0.10)
            low_sat_gate = torch.sigmoid((0.35 - saturation) / 0.08)
            low_contrast_gate = torch.sigmoid((0.08 - local_contrast_vis) / 0.03)
            haze_app = self._smooth01(
                bright_gate * low_sat_gate * low_contrast_gate)

            struct_deg = self._smooth01((1.0 - attn_deg).clamp(0.0, 1.0))

            edge_vis = self._robust_norm(self._sobel_edge(gray_vis))
            edge_ir = self._robust_norm(self._sobel_edge(gray_ir))
            contrast_vis = self._robust_norm(local_contrast_vis)
            contrast_ir = self._robust_norm(local_contrast_ir)

            edge_adv = F.relu(edge_ir - edge_vis)
            contrast_adv = F.relu(contrast_ir - contrast_vis)
            ir_adv = self._smooth01(
                (0.7 * edge_adv + 0.3 * contrast_adv).clamp(0.0, 1.0))

            evidence = 0.4 * struct_deg + 0.6 * ir_adv
            P_pseudo = self._smooth01(haze_app * evidence)
            P_pseudo = P_pseudo.clamp(0.0, 1.0) ** self.gamma
            P_pseudo = P_pseudo.detach()

        # ================================================================
        # (f) Decoder
        # ================================================================
        dec_input = torch.cat([f_vis, feat_clip, feat_dino, P_pseudo], dim=1)
        P_fail = self.decoder(dec_input)

        tau_raw = self.thr_head(dec_input)
        tau = self.tau_min + (self.tau_max - self.tau_min) * tau_raw

        G_dec = torch.sigmoid((P_fail - tau) / self.gate_temperature)
        P_support = torch.clamp(P_pseudo.detach(), 0.0, 1.0) ** self.support_gamma
        G_soft = torch.clamp(G_dec * P_support, 0.0, 1.0)
        M_hard = (G_soft >= self.hard_gate_threshold).float()
        haze_mask = M_hard.detach() + G_soft - G_soft.detach()

        if return_debug:
            return {
                "P_fail": P_fail,
                "P_pseudo": P_pseudo,
                "tau": tau,
                "G_dec": G_dec,
                "P_support": P_support,
                "G_soft": G_soft,
                "M_hard": M_hard,
                "haze_mask": haze_mask,
                "g_fog": g_fog,
                "attn_deg": attn_deg,
                "haze_app": haze_app,
                "struct_deg": struct_deg,
                "ir_adv": ir_adv,
                "edge_vis": edge_vis,
                "edge_ir": edge_ir,
                "contrast_vis": contrast_vis,
                "contrast_ir": contrast_ir,
                "disc": ir_adv,
                "disc_refined": P_pseudo,
            }
        return P_fail, P_pseudo, tau, G_dec, P_support, G_soft, M_hard, haze_mask


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[cmdn] Device: {device}")
    m = CMDN().to(device)
    x = torch.randn(1, 3, 256, 256).to(device)
    y = torch.randn(1, 3, 256, 256).to(device)
    P_fail, P_pseudo, tau, G_dec, P_support, G_soft, M_hard, haze_mask = m(x, y)
    print(f"P_fail: {P_fail.shape}, P_pseudo: {P_pseudo.shape}, tau: {tau.shape}")
    dbg = m(x, y, return_debug=True)
    print(
        f"Debug: g_fog={dbg['g_fog'].shape}, haze_app={dbg['haze_app'].shape}, "
        f"struct_deg={dbg['struct_deg'].shape}, ir_adv={dbg['ir_adv'].shape}, "
        f"P_pseudo_requires_grad={dbg['P_pseudo'].requires_grad}"
    )
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    frozen   = sum(p.numel() for p in m.parameters() if not p.requires_grad)
    print(f"Trainable: {trainable:,}  Frozen: {frozen:,}")
    print("[cmdn] Smoke test PASSED")
