# -*- coding: utf-8 -*-
"""
Haze Mask Auto-Estimation Module — Standalone Verification Script

Usage:
  python verify_haze_mask.py --vis <hazy_vis.jpg> --ir <infrared.jpg> [--output ./verify_output]

If no images provided, only Level 3 (gradient flow) runs on synthetic data.
"""

import argparse
import os
import sys
import io

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
# 0. Pure functions — identical to Teacher.py
# ============================================================================

def compute_haze_density(x_vis_01):
    """Estimate per-pixel haze density from VIS physical cues.

    H = brightness * (1 - saturation) * (1 - local_contrast)
    Higher H -> denser haze.
    """
    L = 0.299 * x_vis_01[:, 0:1, :, :] + \
        0.587 * x_vis_01[:, 1:2, :, :] + \
        0.114 * x_vis_01[:, 2:3, :, :]

    max_rgb = x_vis_01.max(dim=1, keepdim=True)[0]
    min_rgb = x_vis_01.min(dim=1, keepdim=True)[0]
    S = 1.0 - min_rgb / (max_rgb + 1e-6)

    kernel_size = 15
    padding = kernel_size // 2
    kernel = torch.ones(1, 1, kernel_size, kernel_size,
                        device=x_vis_01.device,
                        dtype=x_vis_01.dtype) / (kernel_size ** 2)
    L_mean = F.conv2d(L, kernel, padding=padding)
    L_sq_mean = F.conv2d(L ** 2, kernel, padding=padding)
    C = torch.sqrt((L_sq_mean - L_mean ** 2).clamp(min=0) + 1e-6)

    H = L * (1.0 - S) * (1.0 - C)

    B = H.shape[0]
    flat = H.view(B, -1)
    min_val = flat.min(dim=1)[0].view(B, 1, 1, 1)
    max_val = flat.max(dim=1)[0].view(B, 1, 1, 1)
    H = (H - min_val) / (max_val - min_val + 1e-6)
    return H


def compute_ir_structure(x_ir):
    """Compute per-pixel IR local structure energy, normalized to [0, 1]."""
    C = x_ir.shape[1]
    sobel_x = torch.tensor(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
        dtype=x_ir.dtype, device=x_ir.device
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
        dtype=x_ir.dtype, device=x_ir.device
    ).view(1, 1, 3, 3)
    sobel_x_mc = sobel_x.repeat(C, 1, 1, 1)
    sobel_y_mc = sobel_y.repeat(C, 1, 1, 1)
    Ix = F.conv2d(x_ir, sobel_x_mc, padding=1, groups=C)
    Iy = F.conv2d(x_ir, sobel_y_mc, padding=1, groups=C)
    grad_energy = (Ix ** 2 + Iy ** 2).mean(dim=1, keepdim=True)

    radius = 3
    kernel_size = 7
    coords = torch.arange(kernel_size, device=x_ir.device, dtype=x_ir.dtype) - radius
    sigma = 1.5
    kernel = torch.exp(-(coords.view(kernel_size, 1) ** 2 +
                         coords.view(1, kernel_size) ** 2) / (2 * sigma ** 2))
    kernel = kernel / (kernel.sum() + 1e-6)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    E_ir = F.conv2d(grad_energy, kernel, padding=radius)

    B = E_ir.shape[0]
    flat = E_ir.view(B, -1)
    min_val = flat.min(dim=1)[0].view(B, 1, 1, 1)
    max_val = flat.max(dim=1)[0].view(B, 1, 1, 1)
    return (E_ir - min_val) / (max_val - min_val + 1e-6)


def compute_sky_mask(x_ir):
    B, _, H_s, W_s = x_ir.shape
    y_coords = torch.linspace(0.0, 1.0, H_s, device=x_ir.device, dtype=x_ir.dtype)
    sky_mask = (y_coords < 0.10).view(1, 1, H_s, 1).float().expand(B, 1, H_s, W_s).contiguous()
    return sky_mask


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


# ============================================================================
# 1. Learnable sub-networks
# ============================================================================

class QVisRefine(nn.Module):
    """VIS quality estimator: 5ch -> 1ch."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, padding=1),
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


class HazeMaskEstimator(nn.Module):
    """Complete haze mask estimator with sky exclusion."""
    def __init__(self):
        super().__init__()
        self.q_vis_refine = QVisRefine()

        self.register_buffer(
            'clip_mean',
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'clip_std',
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        )

    def forward(self, x_vis, x_ir, return_all=False):
        # Step 1: Inverse CLIP normalization -> [0,1]
        x_vis_01 = (x_vis * self.clip_std + self.clip_mean).clamp(0, 1)

        # Step 2: Haze density + IR structure + sky exclusion
        H = compute_haze_density(x_vis_01)
        E_ir = compute_ir_structure(x_ir)

        sky_mask = compute_sky_mask(x_ir)
        H_calibrated = H * (1.0 - sky_mask)

        # Step 3: CNN input (5ch: H_calibrated + E_ir + VIS_01)
        vis_input = torch.cat([H_calibrated, E_ir, x_vis_01], dim=1)
        q_vis = self.q_vis_refine(vis_input)

        # Step 4: VIS failure score
        q_complete = 1.0 - q_vis

        # Step 5: Differentiable Otsu + STE binarization
        tau = differentiable_otsu(q_complete)
        m_hard = (q_complete >= tau).float()
        m_soft = torch.sigmoid((q_complete - tau) / 0.1)
        haze_mask = m_hard.detach() + m_soft - m_soft.detach()

        if return_all:
            return {
                'x_vis_01': x_vis_01,
                'H': H,
                'sky_mask': sky_mask,
                'H_calibrated': H_calibrated,
                'E_ir': E_ir,
                'H_x_E_ir': H_calibrated * E_ir,
                'q_vis': q_vis,
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
    """Build master overview figure: 3 rows x 5 columns.

    Row 1: [Input VIS] [H]       [sky_mask] [H_calibr] [E_ir]
    Row 2: [q_vis]     [q_compl] [Otsu]     [m_soft]   [blank]
    Row 3: [m_hard]    [haze_mask] [Overlay VIS] [Overlay IR] [Legend]
    """
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    fig.suptitle('Haze Mask Auto-Estimation — Full Intermediate Variable Overview',
                 fontsize=16, fontweight='bold')

    d = intermediates

    # Row 1: Inputs + haze-density prior + sky exclusion
    _imshow(axes[0, 0], vis_img, '1) Input VIS (hazy)')
    _imshow(axes[0, 1], d['H'],
            '2) H Haze Density\n(bright = dense haze)', cmap='inferno')
    _imshow(axes[0, 2], d['sky_mask'],
            '3) sky_mask (IR flat)\n(white = excluded)', cmap='gray')
    _imshow(axes[0, 3], d['H_calibrated'],
            '4) H_calibrated = H*(1-sky)\n(sky regions suppressed)', cmap='inferno')
    _imshow(axes[0, 4], d['E_ir'],
            '5) E_ir IR Structure\n(bright = rich IR structure)', cmap='inferno')

    # Row 2: Quality estimation + combination
    _imshow(axes[1, 0], d['q_vis'], '6) q_vis VIS Quality\n(bright = clear, learnable)', cmap='viridis')
    _imshow(axes[1, 1], d['q_complete'], '7) q_complete\n= 1 - q_vis', cmap='hot')
    _plot_otsu_histogram(axes[1, 2], d['q_complete'], d['tau'],
                         '8) Otsu Threshold tau\n(soft histogram selection)')
    _imshow(axes[1, 3], d['m_soft'], '9) m_soft\nSigmoid soft mask', cmap='coolwarm')
    axes[1, 4].axis('off')

    # Row 3: Final outputs + overlays
    _imshow(axes[2, 0], d['m_hard'], '10) m_hard Binary Mask\n(q >= tau ? 1 : 0)', cmap='gray')
    _imshow(axes[2, 1], d['haze_mask'], '11) haze_mask (STE)\nDifferentiable binary mask', cmap='gray')
    _overlay(axes[2, 2], vis_img, d['haze_mask'], '12) Mask Overlay on VIS\n(red = detected as haze)')
    _overlay(axes[2, 3], ir_img, d['haze_mask'], '13) Mask Overlay on IR\n(red = needs IR help)')
    _legend(axes[2, 4], d['tau'])

    plt.tight_layout()
    return fig


def _imshow(ax, img, title, cmap=None):
    """Display image on a subplot axis."""
    if isinstance(img, torch.Tensor):
        img = tensor_to_numpy(img)
    if img.ndim == 3 and img.shape[2] == 3:
        ax.imshow(img, aspect='auto')
    else:
        ax.imshow(img, cmap=cmap or 'viridis', aspect='auto')
    ax.set_title(title, fontsize=8)
    ax.axis('off')


def _overlay(ax, bg_img, mask, title):
    """Overlay mask on background (red highlight = positive)."""
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
    overlay[:, :, 0] = np.clip(bg[:, :, 0] + m * 0.6, 0, 1)
    overlay[:, :, 1] = np.clip(bg[:, :, 1] * (1 - m * 0.5), 0, 1)
    overlay[:, :, 2] = np.clip(bg[:, :, 2] * (1 - m * 0.5), 0, 1)

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
        "H:         VIS haze density (brightness\n"
        "           * (1-saturation) * (1-contrast))\n"
        "sky_mask:  IR flat-region mask\n"
        "           (1=sky to exclude)\n"
        "H_calibr:  H * (1 - sky_mask)\n"
        "           sky areas suppressed\n"
        "E_ir:      IR local structure energy\n"
        "q_vis:     VIS pixel quality (learnable)\n"
        "q_complete: 1 - q_vis\n"
        "           -> needs IR intervention\n"
        f"Otsu tau:  {tau_val:.4f}\n"
        "haze_mask: STE diff. binary mask\n"
        "===================================\n"
        "VERIFICATION CHECKLIST\n"
        "1) H: bright where haze is dense?\n"
        "2) sky_mask: excludes flat IR regions?\n"
        "3) H_calibrated: sky suppressed?\n"
        "4) q_vis / haze_mask: marks actual haze?\n"
        "5) Random weights: any effect?\n"
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
    """Verify gradients flow from haze_mask back to learnable params."""
    print("\n" + "=" * 60)
    print("Level 3 — Gradient Flow Verification (Trainability Check)")
    print("=" * 60)

    model.train()
    x_vis.requires_grad = False
    x_ir.requires_grad = False

    intermediates = model(x_vis, x_ir, return_all=True)
    haze_mask = intermediates['haze_mask']

    pseudo_loss = haze_mask.mean()
    pseudo_loss.backward()

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
        for n in params_without_grad:
            print(f"    - {n}")
        print("  [WARN] Unexpected zero-gradient params — check computation graph.")

    cnn_params = trainable_params
    if params_with_grad >= cnn_params:
        print(f"\n  [OK] {params_with_grad}/{cnn_params} CNN params received gradients!")
        print("  => Module is end-to-end trainable.")

    for name, param in model.named_parameters():
        if 'q_vis_refine' in name and param.grad is not None:
            grad_norm = param.grad.abs().mean().item()
            print(f"\n  {name} mean |grad|: {grad_norm:.6e}")
            break

    H = intermediates['H']
    print(f"  H.grad_fn: {H.grad_fn}")

    h_grad_fn = haze_mask.grad_fn
    print(f"  haze_mask.grad_fn: {h_grad_fn}")


# ============================================================================
# 4. Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Verify Haze Mask Auto-Estimation Module')
    parser.add_argument('--vis', type=str, default=None, help='Path to hazy visible image')
    parser.add_argument('--ir', type=str, default=None, help='Path to infrared image')
    parser.add_argument('--output', type=str, default='./verify_output', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    transform = Compose([
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])

    if args.vis and args.ir:
        print(f"\nLoading images:")
        print(f"  VIS: {args.vis}")
        print(f"  IR:  {args.ir}")

        vis_pil = Image.open(args.vis).convert('RGB')
        ir_pil = Image.open(args.ir).convert('RGB')

        orig_h, orig_w = vis_pil.height, vis_pil.width
        print(f"  Original size: {orig_w}x{orig_h}")

        target_h = max(((orig_h // 16) * 16), 16)
        target_w = max(((orig_w // 16) * 16), 16)
        if target_h != orig_h or target_w != orig_w:
            vis_pil = vis_pil.resize((target_w, target_h), Image.BICUBIC)
            ir_pil = ir_pil.resize((target_w, target_h), Image.BICUBIC)
            print(f"  Resized to: {target_w}x{target_h}")

        x_vis = transform(vis_pil).unsqueeze(0).to(device)
        x_ir = transform(ir_pil).unsqueeze(0).to(device)

        vis_np = np.array(vis_pil)
        ir_np = np.array(ir_pil)

        # --- Level 1: Pure haze-density prior (no learnable params) ---
        print("\n" + "=" * 60)
        print("Level 1 — Haze Density Prior Verification (no training)")
        print("=" * 60)
        print("  Computing H, sky_mask, H_calibrated, and E_ir... (no neural network)")

        with torch.no_grad():
            x_vis_01 = (x_vis * torch.tensor([0.26862954, 0.26130258, 0.27577711],
                                              device=device).view(1, 3, 1, 1) +
                        torch.tensor([0.48145466, 0.4578275, 0.40821073],
                                     device=device).view(1, 3, 1, 1)).clamp(0, 1)
            H = compute_haze_density(x_vis_01)
            E_ir = compute_ir_structure(x_ir)
            sky_mask = compute_sky_mask(x_ir)
            H_calibrated = H * (1.0 - sky_mask)

        sky_coverage = sky_mask.mean().item() * 100
        print(f"  H stats:                mean={H.mean().item():.4f}, std={H.std().item():.4f}")
        print(f"  sky_mask coverage:      {sky_coverage:.1f}%")
        print(f"  H_calibrated stats:     mean={H_calibrated.mean().item():.4f}, "
              f"std={H_calibrated.std().item():.4f}")
        print(f"  E_ir stats:             mean={E_ir.mean().item():.4f}, std={E_ir.std().item():.4f}")

        fig1, axes1 = plt.subplots(1, 4, figsize=(16, 4))
        _imshow(axes1[0], vis_np, 'VIS Hazy Input')
        _imshow(axes1[1], H, 'H Haze Density\n(bright = dense haze)', cmap='inferno')
        _imshow(axes1[2], sky_mask,
                'sky_mask (IR flat regions)\n(white = excluded)', cmap='gray')
        _imshow(axes1[3], H_calibrated,
                'H_calibrated = H*(1-sky)\n(sky regions suppressed)', cmap='inferno')
        fig1.suptitle('Level 1: Haze-Density Prior + Sky Exclusion (No Training)',
                      fontsize=14, fontweight='bold')
        plt.tight_layout()
        level1_path = os.path.join(args.output, 'level1_haze_density_prior.png')
        fig1.savefig(level1_path, dpi=150, bbox_inches='tight')
        plt.close(fig1)
        print(f"  -> Saved to: {level1_path}")

        # --- Level 2: Full pipeline (random weights) ---
        print("\n" + "=" * 60)
        print("Level 2 — Full Pipeline Verification (Random Weights)")
        print("=" * 60)
        print("  NOTE: q_vis_refine weights are RANDOM!")
        print("  This establishes a pre-training baseline.")

        model = HazeMaskEstimator().to(device)
        model.eval()

        with torch.no_grad():
            intermediates = model(x_vis, x_ir, return_all=True)

        print(f"  q_vis stats:       mean={intermediates['q_vis'].mean().item():.4f}, "
              f"std={intermediates['q_vis'].std().item():.4f}")
        print(f"  q_complete mean:   {intermediates['q_complete'].mean().item():.4f}")
        print(f"  Otsu tau:          {intermediates['tau'].item():.4f}")
        haze_cov = intermediates['haze_mask'].mean().item()
        print(f"  haze_mask mean (haze coverage): {haze_cov * 100:.1f}%")

        fig2 = build_figure(vis_np, ir_np, intermediates)
        level2_path = os.path.join(args.output, 'level2_full_pipeline_random_weights.png')
        fig2.savefig(level2_path, dpi=150, bbox_inches='tight')
        plt.close(fig2)
        print(f"  -> Saved to: {level2_path}")

        for key in ['H', 'sky_mask', 'H_calibrated', 'E_ir', 'q_vis', 'q_complete', 'haze_mask']:
            val = intermediates[key]
            np_img = tensor_to_numpy(val)
            save_path = os.path.join(args.output, f'intermediate_{key}.png')
            if np_img.ndim == 2:
                Image.fromarray(np_img).save(save_path)
            else:
                Image.fromarray(np_img).save(save_path)
        print(f"  Individual maps saved to: {args.output}/intermediate_*.png")

        verify_gradient_flow(model, x_vis, x_ir)

    else:
        print("\n" + "=" * 60)
        print("No input images. Running Level 3 on synthetic data.")
        print("=" * 60)
        print()
        print("Tip: python verify_haze_mask.py --vis <hazy.jpg> --ir <ir.jpg>")
        print()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        torch.manual_seed(args.seed)
        B, C, H, W = 1, 3, 256, 256
        x_vis_syn = torch.randn(B, C, H, W, device=device) * 0.5
        x_vis_syn[:, :, :, W // 2:] = x_vis_syn[:, :, :, W // 2:] * 0.3 + 0.8
        x_ir_syn = torch.randn(B, C, H, W, device=device) * 0.5

        model = HazeMaskEstimator().to(device)
        verify_gradient_flow(model, x_vis_syn, x_ir_syn)

        print("\n" + "-" * 40)
        print("Synthetic data quick check (random weights):")
        model.eval()
        with torch.no_grad():
            intermediates = model(x_vis_syn, x_ir_syn, return_all=True)

        left_mask = intermediates['haze_mask'][:, :, :, :W // 2].mean().item()
        right_mask = intermediates['haze_mask'][:, :, :, W // 2:].mean().item()
        left_H = intermediates['H'][:, :, :, :W // 2].mean().item()
        right_H = intermediates['H'][:, :, :, W // 2:].mean().item()
        print(f"  H (haze density):  left (clear) = {left_H:.4f}   right (hazy) = {right_H:.4f}")
        print(f"  haze_mask:         left (clear) = {left_mask:.4f}   right (hazy) = {right_mask:.4f}")
        if right_H > left_H:
            print("  [OK] H correctly identifies right half as hazier")
        else:
            print("  [WARN] H failed to distinguish — check synthetic data construction")

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print("""
How to interpret the results:

  Level 1 (haze density prior):
    - H should be bright where haze is dense.
    - sky_mask marks flat IR regions (white = excluded).
    - H_calibrated = H with sky areas suppressed.

  Level 2 (full pipeline, random weights):
    - q_vis with random weights outputs near-uniform noise.
    - haze_mask at random weights is meaningless.
    - Re-run AFTER training to compare.

  Level 3 (gradient flow):
    - CNN params should receive gradients -> trainable.

Suggested workflow:
  1. python verify_haze_mask.py --vis hazy.jpg --ir ir.jpg
  2. Check level1_haze_density_prior.png -> confirm H, sky_mask, H_calibrated
  3. Train the model (even just a few hundred steps)
  4. Load trained q_vis_refine weights and re-run to compare
""")


if __name__ == '__main__':
    main()
