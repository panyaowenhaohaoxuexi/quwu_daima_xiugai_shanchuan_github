"""
model/cmdn.py - Cross-Modal Diagnostic Network

Supervised CMDN for infrared-guided dehazing diagnostics.

NOTE:
    forward return values changed from the old 8-tuple to
    (C, M, mask_logits). model/Teacher.py still contains the old CMDN
    constructor arguments and 8-tuple unpacking; those call sites must be
    updated in the second integration step.
"""

import os
import sys

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


def differentiable_otsu(q_complete, num_bins=256, delta=0.02, temperature=0.01):
    """Differentiable Otsu threshold for a batch of single-channel maps."""
    b = q_complete.shape[0]
    q = q_complete.reshape(b, -1)
    bins = torch.linspace(0, 1, num_bins, device=q_complete.device, dtype=q_complete.dtype)

    diff = q.unsqueeze(-1) - bins.view(1, 1, num_bins)
    hist = torch.exp(-(diff ** 2) / (2 * delta ** 2)).sum(dim=1)
    hist = hist / (hist.sum(dim=1, keepdim=True) + 1e-6)

    bin_values = bins.view(1, num_bins)
    p1 = torch.cumsum(hist, dim=1)
    mu1 = torch.cumsum(hist * bin_values, dim=1)
    mu_total = mu1[:, -1:]
    p2 = 1 - p1
    mu2 = (mu_total - mu1) / (p2 + 1e-6)
    sigma_b = p1 * p2 * (mu1 / (p1 + 1e-6) - mu2) ** 2

    weights = torch.softmax(sigma_b / temperature, dim=1)
    tau = (weights * bin_values).sum(dim=1).view(b, 1, 1, 1)
    return tau


class TextEncoder(nn.Module):
    """Lightweight wrapper around CLIP text modules for CoA-style prompts."""

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts.type(self.dtype) + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)

        if x.shape[0] == tokenized_prompts.shape[0]:
            x = x[torch.arange(x.shape[0], device=x.device), tokenized_prompts.argmax(dim=-1)]
        else:
            x = x[:, -1, :]
        return x @ self.text_projection


# ---------------------------------------------------------------------------
# CMDN
# ---------------------------------------------------------------------------

class CMDN(nn.Module):
    """Supervised Cross-Modal Diagnostic Network.

    Args:
        haze_prompt_path: path to CoA-style haze/clear prompt embeddings.
        clip_download_root: local CLIP model cache directory.
    """

    def __init__(self,
                 haze_prompt_path="./clip_model/haze_prompt.pth",
                 clip_download_root="./clip_model/"):
        super().__init__()

        # 1. Trainable VIS / IR encoders.
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

        # 2. CLIP normalisation constants.
        self.register_buffer(
            'clip_mean',
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer(
            'clip_std',
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

        # 3. CLIP loading (frozen) + final-block patch token hook.
        try:
            import CLIP.clip as clip
        except Exception as e:
            raise RuntimeError(f"[CMDN] Cannot import CLIP.clip: {e}.")
        self._clip_module = clip

        self._clip_model, _ = clip.load(
            "ViT-B/32",
            device="cpu",
            download_root=clip_download_root,
        )
        for p in self._clip_model.parameters():
            p.requires_grad = False
        self._clip_model.eval()

        self.clip_visual = self._clip_model.visual
        self._clip_patch_tokens = None
        self.clip_visual.transformer.resblocks[-1].register_forward_hook(self._clip_hook)

        # 4. CoA-style text prompt anchors for M_d.
        t_haze, t_clear = self._load_prompt_anchors(haze_prompt_path)
        self.register_buffer('t_haze', t_haze)
        self.register_buffer('t_clear', t_clear)

        # 5. Supervised three-way fusion head.
        self.fuse = nn.Sequential(
            _conv3x3_bn_relu(65, 64),
            _conv3x3_bn_relu(64, 32),
        )
        self.head_density = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())
        self.head_mask = nn.Conv2d(32, 1, 1)

    def _clip_hook(self, module, inp, out):
        self._clip_patch_tokens = out

    def _load_prompt_anchors(self, haze_prompt_path):
        """Load haze/clear text features from an existing CoA prompt file."""
        try:
            if not os.path.isfile(haze_prompt_path):
                raise FileNotFoundError(haze_prompt_path)

            data = torch.load(haze_prompt_path, map_location="cpu")
            if isinstance(data, dict):
                data = {k[7:] if k.startswith("module.") else k: v for k, v in data.items()}
            embedding_prompt = data["embedding_prompt"]
            embedding_prompt = nn.Parameter(embedding_prompt.float(), requires_grad=False)

            B_prompt = embedding_prompt.shape[0]
            if B_prompt < 2:
                raise ValueError(f"embedding_prompt must contain haze and clear prompts, got {B_prompt}")

            token_str = " ".join(["X"] * 16)
            tokenized_prompts = torch.cat(
                [self._clip_module.tokenize(token_str) for _ in range(B_prompt)],
                dim=0,
            )
            text_encoder = TextEncoder(self._clip_model)
            text_encoder.eval()

            with torch.no_grad():
                text_features = text_encoder(embedding_prompt, tokenized_prompts).float()
                mid = B_prompt // 2
                if mid == 0 or mid == B_prompt:
                    raise ValueError(f"invalid haze/clear prompt split for B_prompt={B_prompt}")
                t_haze = text_features[:mid].mean(0).float()
                t_clear = text_features[mid:].mean(0).float()
            return t_haze, t_clear
        except Exception as e:
            print(f"[CMDN] Warning: failed to load haze prompts from {haze_prompt_path}: {e}. "
                  "M_d will fall back to constant 0.5.")
            return None, None

    def forward(self, x_vis_clipnorm, x_ir, return_debug=False):
        B, _, H, W = x_vis_clipnorm.shape

        # 1. CLIP semantic density prior M_d.
        x_vis_01 = (x_vis_clipnorm * self.clip_std + self.clip_mean).clamp(0, 1)
        x_vis_224 = F.interpolate(x_vis_01, size=(224, 224), mode='bilinear', align_corners=False)

        clip_dtype = self.clip_visual.conv1.weight.dtype
        _ = self.clip_visual(x_vis_224.type(clip_dtype))   # trigger hook

        raw = self._clip_patch_tokens                      # (50, B, 768)
        patch = raw.to(x_vis_clipnorm.device).permute(1, 0, 2)[:, 1:, :].float()
        proj = self.clip_visual.proj.to(x_vis_clipnorm.device).float()
        patch_512 = patch @ proj

        if self.t_haze is not None and self.t_clear is not None:
            patch_n = F.normalize(patch_512, dim=-1)
            t_haze_n = F.normalize(
                self.t_haze.to(x_vis_clipnorm.device).float().view(1, 1, -1),
                dim=-1,
            )
            t_clear_n = F.normalize(
                self.t_clear.to(x_vis_clipnorm.device).float().view(1, 1, -1),
                dim=-1,
            )
            sim_haze = (patch_n * t_haze_n).sum(-1)
            sim_clear = (patch_n * t_clear_n).sum(-1)
            temperature = 0.01
            sims = torch.stack([sim_haze, sim_clear], dim=1) / temperature
            # [FIXED] haze/clear 方向：经真实浓雾图验证，浓度=第1分量
            M_d_flat = torch.softmax(sims, dim=1)[:, 1, :]
            M_d_small = M_d_flat.reshape(B, 1, 7, 7)
        else:
            M_d_small = torch.full((B, 1, 7, 7), 0.5, device=x_vis_clipnorm.device)

        M_d = F.interpolate(M_d_small, size=(H, W), mode='bilinear', align_corners=False)

        # 2. Trainable VIS / IR features.
        vis_feat = self.vis_enc(x_vis_clipnorm)
        ir_feat = self.ir_enc(x_ir)

        # 3. Three-way fusion.
        fuse_input = torch.cat([M_d, vis_feat, ir_feat], dim=1)
        assert fuse_input.shape[1] == 65
        fused = self.fuse(fuse_input)

        # 4. Supervised outputs.
        C = self.head_density(fused)
        mask_logits = self.head_mask(fused)
        M_prob = torch.sigmoid(mask_logits)

        # 5. Inference-time binary mask with STE gradients.
        tau = differentiable_otsu(M_prob)
        m_hard = (M_prob >= tau).float()
        M = m_hard.detach() + M_prob - M_prob.detach()

        if return_debug:
            return {
                "M_d": M_d,
                "C": C,
                "mask_logits": mask_logits,
                "M_prob": M_prob,
                "tau": tau,
                "m_hard": m_hard,
                "M": M,
            }
        return C, M, mask_logits


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from model.diagnose_loss import density_loss, mask_loss

    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = CMDN().to(device)
    xv = torch.randn(2, 3, 256, 256).to(device)
    xi = torch.randn(2, 3, 256, 256).to(device)
    C, M, logits = m(xv, xi)
    assert C.shape == (2, 1, 256, 256) and M.shape == (2, 1, 256, 256)
    assert C.min() >= 0 and C.max() <= 1
    density_gt = torch.rand(2, 1, 256, 256).to(device)
    mask_gt = (torch.rand(2, 1, 256, 256) > 0.5).float().to(device)
    loss = density_loss(C, density_gt) + mask_loss(logits, mask_gt)
    loss.backward()
    n_grad = sum(p.grad is not None for p in m.parameters() if p.requires_grad)
    clip_frozen = all(not p.requires_grad for p in m._clip_model.parameters())
    required_modules = [m.vis_enc, m.ir_enc, m.fuse, m.head_density, m.head_mask]
    assert clip_frozen
    assert all(any(p.grad is not None for p in module.parameters() if p.requires_grad)
               for module in required_modules)
    print(f"[CMDN] forward OK, C{tuple(C.shape)} M{tuple(M.shape)}, params with grad: {n_grad}")
    print("[CMDN] smoke test PASSED")
