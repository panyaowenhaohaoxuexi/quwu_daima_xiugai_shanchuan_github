# -*- coding: utf-8 -*-
"""
Haze Mask Auto-Estimation Module — Standalone Verification Script

Verification targets:
  Is the newly added haze-aware module actually useful?

Strategy (3 progressive levels):
  Level 1: Does the physics prior t_hat already distinguish haze regions?  (no training)
  Level 2: Full pipeline behavior under random weights                       (baseline)
  Level 3: Can gradients flow back to q_vis_refine / q_ir_estimator?       (trainability)

Usage:
  python verify_haze_mask.py --vis <hazy_vis.jpg> --ir <infrared.jpg> [--output ./verify_output]

You need a pair of: hazy visible image + infrared image (same scene, same resolution).
If no images provided, only Level 3 (gradient flow) runs on synthetic data.
"""

import argparse
import math
import os
import sys
import io

# Fix Windows GBK encoding issue
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize


# ============================================================================
# 0. Pure functions copied from Teacher.py (fully standalone, no model weights)
# ============================================================================

def compute_gaussian_blur(x, sigma):
    """Channel-wise Gaussian blur with a kernel derived from sigma."""
    radius = int(math.ceil(3 * sigma))
    kernel_size = 2 * radius + 1
    coords = torch.arange(kernel_size, device=x.device, dtype=x.dtype) - radius
    yy = coords.view(kernel_size, 1)
    xx = coords.view(1, kernel_size)
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / (kernel.sum() + 1e-6)
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x, kernel, padding=radius, groups=x.shape[1])


def differentiable_otsu(q_complete, num_bins=256, delta=0.02, temperature=0.01):
    """Differentiable Otsu threshold via soft histogram + softmax weighting."""
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


def f_inconsistency(x, y):
    """VIFNet inconsistency function."""
    return (1 - x) * (1 - y) + 0.5 * x * y


# ============================================================================
# 1. Learnable sub-networks (structure identical to Teacher.__init__)
# ============================================================================

class QVisRefine(nn.Module):
    """VIS quality estimator: 10ch -> 1ch."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(10, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class QIrEstimator(nn.Module):
    """IR quality estimator: 3ch -> 1ch, with dilated convolutions for larger receptive field."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class HazeMaskEstimator(nn.Module):
    """
    Complete haze mask estimator.

    Components:
      - q_vis_refine: VIS pixel quality (learnable CNN)
      - q_ir_estimator: IR pixel quality (learnable CNN)
      - t_hat: physics-based transmittance (non-learnable, pure math)
      - Differentiable Otsu + STE binarization (non-learnable)
    """
    def __init__(self):
        super().__init__()
        self.q_vis_refine = QVisRefine()
        self.q_ir_estimator = QIrEstimator()

        # CLIP normalization params (from CLIP training)
        self.register_buffer(
            'clip_mean',
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'clip_std',
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        )

    def forward(self, x_vis, x_ir, return_all=False):
        """
        Args:
            x_vis: (B,3,H,W) CLIP-normalized hazy visible image
            x_ir:  (B,3,H,W) CLIP-normalized infrared image
            return_all: if True, return dict of all intermediate variables

        Returns:
            if return_all=False: haze_mask (B,1,H,W)
            if return_all=True:  dict with all intermediate results
        """
        # Step 1: Inverse CLIP normalization -> [0,1]
        x_vis_01 = (x_vis * self.clip_std + self.clip_mean).clamp(0, 1)

        # Step 2: Physics-based transmittance estimation
        i_low = compute_gaussian_blur(x_vis_01, sigma=3.0)               # low-freq
        i_high = x_vis_01 - i_low                                         # high-freq
        a_hat = i_low.mean(dim=[2, 3], keepdim=True)                     # ambient light
        high_norm = torch.norm(i_high, dim=1, keepdim=True)              # ||high||
        low_air_norm = torch.norm(i_low - a_hat, dim=1, keepdim=True)    # ||low - ambient||
        t_hat = high_norm / (high_norm + low_air_norm + 1e-6)            # transmittance

        # Step 3: Concatenated input -> q_vis_refine
        vis_input = torch.cat([x_vis_01, i_high, i_low, t_hat], dim=1)  # 10ch
        q_vis = self.q_vis_refine(vis_input)                              # (B,1,H,W)

        # Step 4: q_ir_estimator
        q_ir = self.q_ir_estimator(x_ir)                                  # (B,1,H,W)

        # Step 5: Combine: VIS is bad AND IR is good
        q_complete = (1.0 - q_vis) * q_ir

        # Step 6: Differentiable Otsu + STE binarization
        tau = differentiable_otsu(q_complete)
        m_hard = (q_complete >= tau).float()
        m_soft = torch.sigmoid((q_complete - tau) / 0.1)
        haze_mask = m_hard.detach() + m_soft - m_soft.detach()  # STE

        if return_all:
            return {
                'x_vis_01': x_vis_01,
                'i_low': i_low,
                'i_high': i_high,
                't_hat': t_hat,
                'q_vis': q_vis,
                'q_ir': q_ir,
                'q_complete': q_complete,
                'tau': tau,
                'm_hard': m_hard,
                'm_soft': m_soft,
                'haze_mask': haze_mask,
            }
        return haze_mask


# ============================================================================
# 2. Visualization utilities
# ============================================================================

def tensor_to_numpy(t, squeeze_ch=True):
    """(1,C,H,W) -> (H,W) or (C,H,W) numpy uint8"""
    t = t.detach().cpu().squeeze(0)
    if squeeze_ch and t.ndim == 3 and t.shape[0] == 1:
        t = t.squeeze(0)
    t = t.clamp(0, 1) if t.max() > 1.0 else t.clamp(0, 1)
    if t.ndim == 2:
        return (t.numpy() * 255).astype(np.uint8)
    else:
        return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def build_figure(vis_img, ir_img, intermediates):
    """
    Build master overview figure: 3 rows x 5 columns.

    Row 1: [Input VIS] [Input IR]  [t_hat]   [i_low]     [i_high]
    Row 2: [q_vis]      [q_ir]     [q_compl] [Otsu Hist]  [m_soft]
    Row 3: [m_hard]     [haze_mask] [Overlay VIS] [Overlay IR] [Legend]
    """
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle('Haze Mask Auto-Estimation — Full Intermediate Variable Overview',
                 fontsize=16, fontweight='bold')

    d = intermediates

    # Row 1: Inputs + physics layer
    _imshow(axes[0, 0], vis_img, '1) Input VIS (hazy)')
    _imshow(axes[0, 1], ir_img, '2) Input IR')
    _imshow(axes[0, 2], d['t_hat'],
            '3) t_hat Transmittance\n(dark = dense haze, physics-only)', cmap='inferno')
    _imshow(axes[0, 3], d['i_low'],
            '4) i_low Low-freq\n(haze base layer)', cmap='gray')
    _imshow(axes[0, 4], d['i_high'],
            '5) i_high High-freq\n(texture / edges)', cmap='gray')

    # Row 2: Quality estimation + combination
    _imshow(axes[1, 0], d['q_vis'], '6) q_vis VIS Quality\n(bright = clear, learnable)', cmap='viridis')
    _imshow(axes[1, 1], d['q_ir'], '7) q_ir IR Quality\n(bright = strong IR signal, learnable)', cmap='viridis')
    _imshow(axes[1, 2], d['q_complete'], '8) q_complete\n= (1-q_vis) * q_ir', cmap='hot')
    _plot_otsu_histogram(axes[1, 3], d['q_complete'], d['tau'],
                         '9) Otsu Threshold tau\n(soft histogram selection)')
    _imshow(axes[1, 4], d['m_soft'], '10) m_soft\nSigmoid soft mask', cmap='coolwarm')

    # Row 3: Final outputs + overlays
    _imshow(axes[2, 0], d['m_hard'], '11) m_hard Binary Mask\n(q >= tau ? 1 : 0)', cmap='gray')
    _imshow(axes[2, 1], d['haze_mask'], '12) haze_mask (STE)\nDifferentiable binary mask', cmap='gray')
    _overlay(axes[2, 2], vis_img, d['haze_mask'], '13) Mask Overlay on VIS\n(red = detected as haze)')
    _overlay(axes[2, 3], ir_img, d['haze_mask'], '14) Mask Overlay on IR\n(red = needs IR help)')
    _legend(axes[2, 4], d['tau'])

    plt.tight_layout()
    return fig


def _imshow(ax, img, title, cmap=None):
    """Display image on a subplot axis."""
    if isinstance(img, torch.Tensor):
        img = tensor_to_numpy(img)
    if img.ndim == 3 and img.shape[2] == 3:
        ax.imshow(img)
    else:
        ax.imshow(img, cmap=cmap or 'viridis')
    ax.set_title(title, fontsize=8)
    ax.axis('off')


def _overlay(ax, bg_img, mask, title):
    """Overlay mask on background image (red highlight for positives)."""
    if isinstance(bg_img, torch.Tensor):
        bg = tensor_to_numpy(bg_img).astype(np.float32) / 255.0
    else:
        bg = bg_img.astype(np.float32) / 255.0
    if isinstance(mask, torch.Tensor):
        m = tensor_to_numpy(mask).astype(np.float32) / 255.0
    else:
        m = mask.astype(np.float32) / 255.0

    if bg.ndim == 2:
        bg = np.stack([bg, bg, bg], axis=-1)
    if m.ndim == 3:
        m = m.squeeze(-1) if m.shape[-1] == 1 else m[:, :, 0]

    overlay = bg.copy()
    overlay[:, :, 0] = np.clip(bg[:, :, 0] + m * 0.6, 0, 1)   # boost red channel
    overlay[:, :, 1] = np.clip(bg[:, :, 1] * (1 - m * 0.5), 0, 1)  # suppress green
    overlay[:, :, 2] = np.clip(bg[:, :, 2] * (1 - m * 0.5), 0, 1)  # suppress blue

    ax.imshow(overlay)
    ax.set_title(title, fontsize=8)
    ax.axis('off')


def _plot_otsu_histogram(ax, q_complete, tau, title):
    """Plot histogram of q_complete with Otsu threshold line."""
    if isinstance(q_complete, torch.Tensor):
        vals = q_complete.detach().cpu().numpy().ravel()
    else:
        vals = q_complete.ravel()
    if isinstance(tau, torch.Tensor):
        tau_val = tau.detach().cpu().item()
    else:
        tau_val = tau

    ax.hist(vals, bins=128, range=(0, 1), color='steelblue', alpha=0.7, edgecolor='none')
    ax.axvline(x=tau_val, color='red', linestyle='--', linewidth=2,
               label=f'tau = {tau_val:.4f}')
    ax.set_xlim(0, 1)
    ax.set_xlabel('q_complete value')
    ax.set_ylabel('Pixel count')
    ax.set_title(title, fontsize=8)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def _legend(ax, tau):
    """Display text legend with interpretation guide."""
    tau_val = tau.detach().cpu().item() if isinstance(tau, torch.Tensor) else tau
    ax.axis('off')
    text = (
        "LEGEND\n"
        "===================================\n"
        "t_hat:     Physics transmittance est.\n"
        "           No training, plug-and-play\n"
        "q_vis:     VIS pixel quality (learnable)\n"
        "q_ir:      IR pixel quality (learnable)\n"
        "q_complete: VIS_bad AND IR_good\n"
        "           -> needs IR intervention\n"
        f"Otsu tau:  {tau_val:.4f}\n"
        "haze_mask: STE diff. binary mask\n"
        "===================================\n"
        "VERIFICATION CHECKLIST\n"
        "1) t_hat: dark in haze, bright in clear?\n"
        "2) haze_mask: marks actual haze regions?\n"
        "3) Random weights: any effect?\n"
        "   (If yes -> strong physics prior)\n"
        "   (If no  -> training is needed)\n"
        "===================================\n"
        "NEXT STEP\n"
        "Re-run after training to compare!"
    )
    ax.text(0.02, 0.98, text, transform=ax.transAxes,
            fontsize=7, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.85))


# ============================================================================
# 3. Gradient flow verification
# ============================================================================

def verify_gradient_flow(model, x_vis, x_ir):
    """
    Verify that gradients can flow from haze_mask back to learnable params.

    Uses the mean of haze_mask as a pseudo-loss, then checks whether
    q_vis_refine and q_ir_estimator params receive non-zero gradients.
    """
    print("\n" + "=" * 60)
    print("Level 3 — Gradient Flow Verification (Trainability Check)")
    print("=" * 60)

    model.train()
    x_vis.requires_grad = False
    x_ir.requires_grad = False

    # Forward
    intermediates = model(x_vis, x_ir, return_all=True)
    haze_mask = intermediates['haze_mask']

    # Pseudo loss: minimize haze_mask mean (simulates "network adjusts the mask")
    pseudo_loss = haze_mask.mean()
    pseudo_loss.backward()

    # Check gradients on learnable params
    trainable_params = 0
    params_with_grad = 0
    params_without_grad = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += 1
            if param.grad is not None and param.grad.abs().sum() > 0:
                params_with_grad += 1
            else:
                params_without_grad.append(name)

    print(f"  Total trainable param groups: {trainable_params}")
    print(f"  Received non-zero gradient:   {params_with_grad}")
    print(f"  Received zero gradient:       {len(params_without_grad)}")

    if params_without_grad:
        print("\n  [WARN] Params with zero gradient:")
        for n in params_without_grad:
            print(f"    - {n}")

    if params_with_grad == trainable_params:
        print("\n  [OK] All learnable parameters received gradients!")
        print("  => Module is end-to-end trainable.")
    else:
        print(f"\n  [WARN] Only {params_with_grad}/{trainable_params} received gradients.")
        print("  => Possible STE bug or broken computation graph.")

    # Check gradient on q_vis_refine specifically
    for name, param in model.named_parameters():
        if 'q_vis_refine' in name and param.grad is not None:
            grad_norm = param.grad.abs().mean().item()
            print(f"\n  {name} mean |grad|: {grad_norm:.6e}")
            break

    # t_hat should NOT have grad_fn (pure math, no parameters)
    t_hat = intermediates['t_hat']
    if t_hat.grad_fn is None:
        print("  t_hat grad_fn: None -> physics branch excluded from backward [OK]")
    else:
        print("  t_hat has grad_fn -> physics branch unexpectedly in computation graph")

    # STE working?
    h_grad_fn = haze_mask.grad_fn
    print(f"  haze_mask.grad_fn: {h_grad_fn}")


# ============================================================================
# 4. Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Verify Haze Mask Auto-Estimation Module')
    parser.add_argument('--vis', type=str, default=None,
                        help='Path to hazy visible image')
    parser.add_argument('--ir', type=str, default=None,
                        help='Path to infrared image')
    parser.add_argument('--output', type=str, default='./verify_output',
                        help='Output directory (default: ./verify_output)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # CLIP normalization (same as training)
    transform = Compose([
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])

    # =====================================================================
    # Level 1 & 2: require real images
    # =====================================================================
    if args.vis and args.ir:
        print(f"\nLoading images:")
        print(f"  VIS: {args.vis}")
        print(f"  IR:  {args.ir}")

        vis_pil = Image.open(args.vis).convert('RGB')
        ir_pil = Image.open(args.ir).convert('RGB')

        orig_h, orig_w = vis_pil.height, vis_pil.width
        print(f"  Original size: {orig_w}x{orig_h}")

        # Resize to multiple of 16
        target_h = max(((orig_h // 16) * 16), 16)
        target_w = max(((orig_w // 16) * 16), 16)
        if target_h != orig_h or target_w != orig_w:
            vis_pil = vis_pil.resize((target_w, target_h), Image.BICUBIC)
            ir_pil = ir_pil.resize((target_w, target_h), Image.BICUBIC)
            print(f"  Resized to: {target_w}x{target_h}")

        x_vis = transform(vis_pil).unsqueeze(0).to(device)  # (1,3,H,W)
        x_ir = transform(ir_pil).unsqueeze(0).to(device)

        vis_np = np.array(vis_pil)
        ir_np = np.array(ir_pil)

        # --- Level 1: Pure physics prior t_hat (no learnable params) ---
        print("\n" + "=" * 60)
        print("Level 1 — Physics Prior Verification (no training)")
        print("=" * 60)
        print("  Computing t_hat (transmittance)... (no neural network involved)")

        with torch.no_grad():
            x_vis_01 = (x_vis * torch.tensor([0.26862954, 0.26130258, 0.27577711],
                                              device=device).view(1, 3, 1, 1) +
                        torch.tensor([0.48145466, 0.4578275, 0.40821073],
                                     device=device).view(1, 3, 1, 1)).clamp(0, 1)
            i_low = compute_gaussian_blur(x_vis_01, sigma=3.0)
            i_high = x_vis_01 - i_low
            a_hat = i_low.mean(dim=[2, 3], keepdim=True)
            high_norm = torch.norm(i_high, dim=1, keepdim=True)
            low_air_norm = torch.norm(i_low - a_hat, dim=1, keepdim=True)
            t_hat = high_norm / (high_norm + low_air_norm + 1e-6)

        t_hat_mean = t_hat.mean().item()
        t_hat_std = t_hat.std().item()
        print(f"  t_hat stats:     mean={t_hat_mean:.4f}, std={t_hat_std:.4f}")
        print(f"  t_hat < 0.2 (dense haze):  {(t_hat < 0.2).float().mean().item() * 100:.1f}%")
        print(f"  t_hat > 0.8 (clear area):  {(t_hat > 0.8).float().mean().item() * 100:.1f}%")

        # Save physics prior result
        fig1, axes1 = plt.subplots(1, 4, figsize=(16, 4))
        _imshow(axes1[0], vis_np, 'VIS Hazy Input')
        _imshow(axes1[1], ir_np, 'IR Input')
        _imshow(axes1[2], t_hat,
                't_hat Transmittance\n(dark = dense haze, bright = clear)', cmap='inferno')
        _imshow(axes1[3], i_high.mean(dim=1, keepdim=True),
                'High-freq Texture Strength\n(dark = hazy, physics quantity)', cmap='gray')
        fig1.suptitle('Level 1: Physics Prior — t_hat Transmittance (No Training Needed)',
                      fontsize=14, fontweight='bold')
        plt.tight_layout()
        level1_path = os.path.join(args.output, 'level1_physics_prior.png')
        fig1.savefig(level1_path, dpi=150, bbox_inches='tight')
        plt.close(fig1)
        print(f"  -> Saved to: {level1_path}")

        # --- Level 2: Full pipeline (with untrained random-weight CNNs) ---
        print("\n" + "=" * 60)
        print("Level 2 — Full Pipeline Verification (Random Weights)")
        print("=" * 60)
        print("  NOTE: q_vis_refine and q_ir_estimator weights are RANDOM!")
        print("  This establishes a pre-training baseline.")
        print("  Re-run after training to compare the improvement.")

        model = HazeMaskEstimator().to(device)
        model.eval()

        with torch.no_grad():
            intermediates = model(x_vis, x_ir, return_all=True)

        print(f"  q_vis stats:       mean={intermediates['q_vis'].mean().item():.4f}, "
              f"std={intermediates['q_vis'].std().item():.4f}")
        print(f"  q_ir stats:        mean={intermediates['q_ir'].mean().item():.4f}, "
              f"std={intermediates['q_ir'].std().item():.4f}")
        print(f"  q_complete mean:   {intermediates['q_complete'].mean().item():.4f}")
        print(f"  Otsu tau:          {intermediates['tau'].item():.4f}")
        haze_coverage = intermediates['haze_mask'].mean().item()
        print(f"  haze_mask mean (haze coverage): {haze_coverage * 100:.1f}%")

        fig2 = build_figure(vis_np, ir_np, intermediates)
        level2_path = os.path.join(args.output, 'level2_full_pipeline_random_weights.png')
        fig2.savefig(level2_path, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f"  -> Saved to: {level2_path}")

        # Save individual intermediate maps for detailed inspection
        for key in ['t_hat', 'q_vis', 'q_ir', 'q_complete', 'haze_mask']:
            val = intermediates[key]
            np_img = tensor_to_numpy(val)
            save_path = os.path.join(args.output, f'intermediate_{key}.png')
            if np_img.ndim == 2:
                Image.fromarray(np_img).save(save_path)
            else:
                Image.fromarray(np_img).save(save_path)
        print(f"  Individual intermediate maps saved to: {args.output}/intermediate_*.png")

        # --- Level 3: Gradient flow ---
        verify_gradient_flow(model, x_vis, x_ir)

    else:
        # =================================================================
        # No input images — only run gradient flow check on synthetic data
        # =================================================================
        print("\n" + "=" * 60)
        print("No input images provided. Running Level 3 on synthetic data.")
        print("=" * 60)
        print()
        print("Tip: To run full visual verification, provide a pair of images:")
        print("  python verify_haze_mask.py --vis <hazy_vis.jpg> --ir <infrared.jpg>")
        print()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Generate synthetic hazy image: left half clear, right half hazy
        torch.manual_seed(args.seed)
        B, C, H, W = 1, 3, 256, 256
        x_vis_syn = torch.randn(B, C, H, W, device=device) * 0.5

        # Inject artificial haze in right half (lower contrast + offset)
        # In CLIP space: lower variance + higher mean
        x_vis_syn[:, :, :, W // 2:] = x_vis_syn[:, :, :, W // 2:] * 0.3 + 0.8
        x_ir_syn = torch.randn(B, C, H, W, device=device) * 0.5

        model = HazeMaskEstimator().to(device)
        verify_gradient_flow(model, x_vis_syn, x_ir_syn)

        # Quick check on synthetic data
        print("\n" + "-" * 40)
        print("Synthetic data quick check (random weights):")
        model.eval()
        with torch.no_grad():
            intermediates = model(x_vis_syn, x_ir_syn, return_all=True)

        left_mask = intermediates['haze_mask'][:, :, :, :W // 2].mean().item()
        right_mask = intermediates['haze_mask'][:, :, :, W // 2:].mean().item()
        left_t_hat = intermediates['t_hat'][:, :, :, :W // 2].mean().item()
        right_t_hat = intermediates['t_hat'][:, :, :, W // 2:].mean().item()
        print(f"  t_hat:       left (clear) = {left_t_hat:.4f}   right (hazy) = {right_t_hat:.4f}")
        print(f"  haze_mask:   left (clear) = {left_mask:.4f}   right (hazy) = {right_mask:.4f}")
        if right_t_hat < left_t_hat:
            print("  [OK] t_hat correctly identifies right half as hazier (lower transmittance)")
        else:
            print("  [WARN] t_hat failed to distinguish haze — check synthetic data construction")

    # =====================================================================
    # Final summary
    # =====================================================================
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print("""
How to interpret the results:

  Level 1 (t_hat physics prior):
    - Look at the t_hat map: haze areas should be DARK (low transmittance),
      clear areas should be BRIGHT.
    - If this fails -> physics assumptions don't match your data
      (e.g., haze is uniform or image has no texture variation).
    - If this succeeds -> physics prior is effective, you have a reliable signal.

  Level 2 (full pipeline, random weights):
    - q_vis / q_ir with random weights likely output near-uniform noise.
    - Therefore q_complete and haze_mask are also meaningless random values.
    - This does NOT mean the module is useless! It means training is necessary.
    - The key is to re-run this script AFTER training and compare.

  Level 3 (gradient flow):
    - If all params receive gradients -> module is trainable.
    - If some params have zero gradient -> STE or computation graph has a bug.

Suggested workflow:
  1. Prepare a few representative hazy+IR image pairs
  2. Run: python verify_haze_mask.py --vis hazy.jpg --ir ir.jpg
  3. Check level1_physics_prior.png -> confirm t_hat physics prior works
  4. Record level2 random-weight baseline
  5. Train the model (even just a few hundred steps)
  6. Load trained q_vis_refine / q_ir_estimator weights
  7. Re-run this script -> compare and confirm learning effect
""")


if __name__ == '__main__':
    main()
