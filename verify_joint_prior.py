# -*- coding: utf-8 -*-
"""
Joint Prior Verification Script: J = H × E_ir

验证联合物理先验是否能在不使用任何神经网络、不依赖位置假设的前提下，
自然区分「天空区域」和「浓雾遮挡场景区域」，并输出有意义的二值mask。

Usage:
  python verify_joint_prior.py \
    --vis /path/to/REAL_FOGGY/hazy/00896.png \
    --ir  /path/to/REAL_FOGGY/ir/00896.png \
    --output ./verify_output
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
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize


# ============================================================================
# CLIP normalization constants
# ============================================================================

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


# ============================================================================
# 1. Core physical prior functions (copied from verify_haze_mask.py)
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
    """Compute per-pixel IR local structure via local standard deviation, normalized to [0, 1].

    Uses a 15×15 box-window local standard deviation to capture all spatial variation
    (sharp edges AND gradual thermal gradients), not just Sobel high-frequency edges.
    Truly uniform regions (pure sky, pure water/fog) → LocalStd ≈ 0.
    """
    # 1. RGB -> grayscale
    gray = 0.299 * x_ir[:, 0:1, :, :] + \
           0.587 * x_ir[:, 1:2, :, :] + \
           0.114 * x_ir[:, 2:3, :, :]  # (B, 1, H, W)

    # 2. 15×15 box mean filter
    window_size = 15
    padding = window_size // 2
    kernel = torch.ones(1, 1, window_size, window_size,
                        device=x_ir.device, dtype=x_ir.dtype) / (window_size ** 2)

    # 3. Local mean
    local_mean = F.conv2d(gray, kernel, padding=padding)

    # 4. Local mean of squares
    local_sq_mean = F.conv2d(gray ** 2, kernel, padding=padding)

    # 5. Local variance (numerically stable clamp)
    local_var = (local_sq_mean - local_mean ** 2).clamp(min=0)

    # 6. Local standard deviation
    E_ir = torch.sqrt(local_var + 1e-6)

    # 7. Per-image min-max normalization to [0, 1]
    B = E_ir.shape[0]
    flat = E_ir.view(B, -1)
    min_val = flat.min(dim=1)[0].view(B, 1, 1, 1)
    max_val = flat.max(dim=1)[0].view(B, 1, 1, 1)
    return (E_ir - min_val) / (max_val - min_val + 1e-6)


# ============================================================================
# 2. Joint prior and Otsu thresholding
# ============================================================================

def compute_joint_prior(H, E_ir):
    """Combine haze density and IR structure into joint prior.

    J = H × E_ir

    Physical rationale:
    - Sky:     H high × E_ir low  ≈ 0  (uniform IR, no structure)
    - Dense haze over ground: H high × E_ir high → J high (textured IR)
    - Clear scene: H low           → J low  (no haze at all)
    """
    return H * E_ir


def hard_otsu_threshold(J):
    """Standard non-differentiable Otsu thresholding on 256-bin histogram.

    Args:
        J: (B, 1, H, W) float tensor in [0, 1]

    Returns:
        binary_mask: (B, 1, H, W) float tensor, values 0 or 1
        best_tau:    scalar float, the Otsu threshold
    """
    B = J.shape[0]
    binary_masks = []
    best_taus = []

    for b in range(B):
        vals = J[b].reshape(-1)  # flatten to 1D

        # Map [0, 1] -> [0, 255] integer bins
        vals_255 = (vals * 255).clamp(0, 255).long()
        hist = torch.zeros(256, device=J.device, dtype=torch.float32)
        hist.scatter_add_(0, vals_255, torch.ones_like(vals, dtype=torch.float32))

        total = hist.sum()
        if total == 0:
            binary_masks.append(torch.zeros_like(J[b]))
            best_taus.append(0.0)
            continue

        # Cumulative sums
        bin_centers = torch.arange(256, device=J.device, dtype=torch.float32) / 255.0

        w0 = torch.cumsum(hist, dim=0)                    # foreground weight
        sum0 = torch.cumsum(hist * bin_centers, dim=0)    # foreground weighted sum

        total_sum = sum0[-1]
        total_w = w0[-1]

        w1 = total_w - w0                                 # background weight
        sum1 = total_sum - sum0                           # background weighted sum

        mu0 = sum0 / (w0 + 1e-6)
        mu1 = sum1 / (w1 + 1e-6)

        between_var = w0 * w1 * (mu0 - mu1) ** 2

        # Exclude degenerate bins (w0==0 or w1==0)
        valid = (w0 > 0) & (w1 > 0)
        between_var[~valid] = -1.0

        best_bin = between_var.argmax().item()
        best_tau = best_bin / 255.0

        binary = (J[b:b+1] >= best_tau).float()
        binary_masks.append(binary)
        best_taus.append(best_tau)

    binary_mask = torch.cat(binary_masks, dim=0)
    best_tau_val = best_taus[0]  # return scalar for single-batch case

    return binary_mask, best_tau_val


# ============================================================================
# 3. Visualization utilities
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


def _imshow(ax, img, title, cmap=None):
    """Display image on a subplot axis with mean/std annotation."""
    if isinstance(img, torch.Tensor):
        arr = img.detach().cpu()
        mean_val = arr.float().mean().item()
        std_val = arr.float().std().item()
        img = tensor_to_numpy(img)
    elif isinstance(img, np.ndarray):
        arr = img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img
        mean_val = arr.mean()
        std_val = arr.std()
    else:
        mean_val = 0.0
        std_val = 0.0

    if img.ndim == 3 and img.shape[2] == 3:
        ax.imshow(img, aspect='auto')
    else:
        im = ax.imshow(img, cmap=cmap or 'viridis', aspect='auto')
    ax.set_title(f'{title}\n(mean={mean_val:.4f}, std={std_val:.4f})', fontsize=8, pad=3)
    ax.axis('off')


def _overlay(ax, bg_img, mask, title):
    """Overlay mask on background image (red semi-transparent = mask positive)."""
    if isinstance(bg_img, torch.Tensor):
        bg = tensor_to_numpy(bg_img).astype(np.float32) / 255.0
    elif bg_img.dtype == np.uint8:
        bg = bg_img.astype(np.float32) / 255.0
    else:
        bg = bg_img.astype(np.float32)

    if isinstance(mask, torch.Tensor):
        m = mask.detach().cpu().squeeze(0).squeeze(0).numpy()
    else:
        m = mask.squeeze()

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


# ============================================================================
# 4. Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Verify Joint Prior J = H × E_ir (no neural network, no position prior)'
    )
    parser.add_argument('--vis', type=str,
                        default=r'F:\Dehaze_Paper\2_Dataset\1_main_benchmark\REAL_FOGGY\hazy\00960.png',
                        help='Path to hazy visible image')
    parser.add_argument('--ir', type=str,
                        default=r'F:\Dehaze_Paper\2_Dataset\1_main_benchmark\REAL_FOGGY\ir\00960.png',
                        help='Path to infrared image')
    parser.add_argument('--output', type=str, default='./verify_output/joint_prior',
                        help='Output directory (default: ./verify_joint_prior)')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Image loading & preprocessing (identical to verify_haze_mask.py)
    # ------------------------------------------------------------------

    transform = Compose([
        ToTensor(),
        Normalize(CLIP_MEAN, CLIP_STD),
    ])

    print(f"\nLoading images:")
    print(f"  VIS: {args.vis}")
    print(f"  IR:  {args.ir}")

    vis_pil = Image.open(args.vis).convert('RGB')
    ir_pil = Image.open(args.ir).convert('RGB')

    orig_h, orig_w = vis_pil.height, vis_pil.width
    print(f"  Original size: {orig_w}x{orig_h}")

    # Resize to multiples of 16 (min 16)
    target_h = max(((orig_h // 16) * 16), 16)
    target_w = max(((orig_w // 16) * 16), 16)
    if target_h != orig_h or target_w != orig_w:
        vis_pil = vis_pil.resize((target_w, target_h), Image.BICUBIC)
        ir_pil = ir_pil.resize((target_w, target_h), Image.BICUBIC)
        print(f"  Resized to: {target_w}x{target_h}")

    # CLIP-normalized tensors
    x_vis = transform(vis_pil).unsqueeze(0).to(device)
    x_ir = transform(ir_pil).unsqueeze(0).to(device)

    # Original numpy images for display (not CLIP-normalized)
    vis_np = np.array(vis_pil)
    ir_np = np.array(ir_pil)

    # ------------------------------------------------------------------
    # Compute physical priors (no neural network, no trainable params)
    # ------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("Joint Prior Verification: J = H × E_ir")
    print("=" * 60)
    print("  Computing H (haze density), E_ir (IR structure), and J...")

    with torch.no_grad():
        # Inverse CLIP normalization -> [0, 1]
        clip_mean_t = torch.tensor(CLIP_MEAN, device=device).view(1, 3, 1, 1)
        clip_std_t = torch.tensor(CLIP_STD, device=device).view(1, 3, 1, 1)
        x_vis_01 = (x_vis * clip_std_t + clip_mean_t).clamp(0, 1)

        # Haze density from VIS
        H = compute_haze_density(x_vis_01)

        # IR structure energy
        E_ir = compute_ir_structure(x_ir)

        # Joint prior
        J = compute_joint_prior(H, E_ir)

        # Hard Otsu thresholding
        binary_mask, tau = hard_otsu_threshold(J)

    # ------------------------------------------------------------------
    # Diagnostic statistics
    # ------------------------------------------------------------------

    mask_bool = binary_mask.bool().squeeze()  # (H, W)
    mask_float = binary_mask.float()

    # Region 1: completion zone (mask=1, needs IR replacement)
    completion_H = H.squeeze()[mask_bool]
    completion_Eir = E_ir.squeeze()[mask_bool]
    completion_J = J.squeeze()[mask_bool]

    # Region 0: fusion zone (mask=0, VIS works fine)
    fusion_H = H.squeeze()[~mask_bool]
    fusion_Eir = E_ir.squeeze()[~mask_bool]
    fusion_J = J.squeeze()[~mask_bool]

    total_pixels = mask_bool.numel()
    completion_pixels = mask_bool.sum().item()
    fusion_pixels = total_pixels - completion_pixels

    completion_H_mean = completion_H.mean().item() if completion_pixels > 0 else 0.0
    completion_Eir_mean = completion_Eir.mean().item() if completion_pixels > 0 else 0.0
    completion_J_mean = completion_J.mean().item() if completion_pixels > 0 else 0.0

    fusion_H_mean = fusion_H.mean().item() if fusion_pixels > 0 else 0.0
    fusion_Eir_mean = fusion_Eir.mean().item() if fusion_pixels > 0 else 0.0
    fusion_J_mean = fusion_J.mean().item() if fusion_pixels > 0 else 0.0

    print("\n" + "-" * 40)
    print("DIAGNOSTIC STATISTICS")
    print("-" * 40)

    print(f"\n  补全区 (mask=1, needs IR replacement):")
    print(f"    Pixel count:  {completion_pixels} ({completion_pixels / total_pixels * 100:.1f}%)")
    print(f"    H  mean:      {completion_H_mean:.4f}")
    print(f"    E_ir mean:    {completion_Eir_mean:.4f}")
    print(f"    J  mean:      {completion_J_mean:.4f}")

    print(f"\n  融合区 (mask=0, VIS works fine):")
    print(f"    Pixel count:  {fusion_pixels} ({fusion_pixels / total_pixels * 100:.1f}%)")
    print(f"    H  mean:      {fusion_H_mean:.4f}")
    print(f"    E_ir mean:    {fusion_Eir_mean:.4f}")
    print(f"    J  mean:      {fusion_J_mean:.4f}")

    print(f"\n  Otsu threshold tau: {tau:.4f}")

    # Sky suppression check
    if fusion_Eir_mean < completion_Eir_mean * 0.5:
        sky_suppression = "PASS"
    else:
        sky_suppression = "WARN"
    print(f"\n  天空抑制是否有效: {sky_suppression}")
    if sky_suppression == "PASS":
        print(f"    -> Fusion zone E_ir ({fusion_Eir_mean:.4f}) is sufficiently lower")
        print(f"       than completion zone E_ir ({completion_Eir_mean:.4f}),")
        print(f"       indicating the joint prior naturally separates sky from dense haze.")
    else:
        print(f"    -> Fusion zone E_ir ({fusion_Eir_mean:.4f}) is NOT sufficiently lower")
        print(f"       than completion zone E_ir ({completion_Eir_mean:.4f}).")
        print(f"       The joint prior may not be separating sky from haze effectively.")

    # ------------------------------------------------------------------
    # Visualization: 2 rows × 4 columns
    # ------------------------------------------------------------------

    print(f"\n  Generating visualization figure...")

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle('Level 1 Joint Prior Verification: J = H × E_ir',
                 fontsize=16, fontweight='bold')
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02,
                        hspace=0.30, wspace=0.12)

    # Row 1
    _imshow(axes[0, 0], vis_np, 'VIS (Original)')
    _imshow(axes[0, 1], ir_np, 'IR (Original)')
    _imshow(axes[0, 2], H,
            'H: Haze Density\n(brightness × (1-sat) × (1-contrast))', cmap='inferno')
    _imshow(axes[0, 3], E_ir,
            'E_ir: IR Local Std\n(local std dev, window=15)', cmap='inferno')

    # Row 2
    _imshow(axes[1, 0], J,
            'J = H × E_ir Joint Prior\n(sky≈0, dense haze→high)', cmap='inferno')
    _imshow(axes[1, 1], binary_mask,
            f'Binary Mask (Otsu tau={tau:.4f})\n(white=completion, black=fusion)', cmap='gray')
    _overlay(axes[1, 2], vis_np, binary_mask,
             'Mask Overlay on VIS\n(red = needs IR replacement)')
    _overlay(axes[1, 3], ir_np, binary_mask,
             'Mask Overlay on IR\n(red = needs IR replacement)')

    save_path = os.path.join(args.output, 'joint_prior_verification.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved to: {save_path}")

    # Also save individual intermediate maps
    for name, tensor in [('H', H), ('E_ir', E_ir), ('J', J), ('binary_mask', binary_mask)]:
        np_img = tensor_to_numpy(tensor)
        indiv_path = os.path.join(args.output, f'joint_prior_{name}.png')
        if np_img.ndim == 2:
            Image.fromarray(np_img).save(indiv_path)
        else:
            Image.fromarray(np_img).save(indiv_path)
    print(f"  Individual maps saved to: {args.output}/joint_prior_*.png")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    print(f"""
  Physical priors computed:
    H  (haze density):        mean={H.mean().item():.4f},  std={H.std().item():.4f}
    E_ir (IR structure):      mean={E_ir.mean().item():.4f}, std={E_ir.std().item():.4f}
    J = H × E_ir (joint):    mean={J.mean().item():.4f},  std={J.std().item():.4f}

  Otsu threshold:
    tau = {tau:.4f}
    Completion zone (mask=1): {completion_pixels / total_pixels * 100:.1f}%
    Fusion zone     (mask=0): {fusion_pixels / total_pixels * 100:.1f}%

  Sky suppression check:
    Fusion zone E_ir  = {fusion_Eir_mean:.4f}
    Completion zone E_ir = {completion_Eir_mean:.4f}
    Result: {sky_suppression}

  Interpretation:
    - H captures haze density from VIS cues
    - E_ir captures structural richness in IR
    - J = H × E_ir: sky (H↑×E_ir↓≈0), dense haze (H↑×E_ir↑→high), clear (H↓→low)
    - Binary mask = Otsu(J): pixels needing IR-based completion
    - No neural network, no position prior, no learnable parameters used.
""")


if __name__ == '__main__':
    main()
