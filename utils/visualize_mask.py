"""
utils/visualize_mask.py

Haze mask visualization tool during training.
Call visualize_epoch_mask() at the end of each epoch,
saves a multi-column comparison image:
  Col1: Hazy visible image (denormalized to [0,1])
  Col2: Infrared image (normalized to [0,1])
  Col3: g_fog — CLIP sliding-window fog density
  Col4: attn_deg — DINOv2 local structure variance
  Col5: disc_refined — fused pseudo-label
  Col6: M_vis — CMDN soft density map
  Col7: Binary haze mask (threshold 0.5)

Saved to {save_dir}/mask_vis/epoch_{epoch:03d}.png each epoch.
Fixed first N samples ensure cross-epoch comparability.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend, safe on headless servers
import matplotlib.pyplot as plt


# CLIP normalization params (consistent with model/Teacher.py)
_CLIP_MEAN = torch.tensor([0.48145466, 0.4578275,  0.40821073]).view(1, 3, 1, 1)
_CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def _to_numpy_rgb(tensor_chw):
    """(C,H,W) float tensor -> (H,W,3) uint8 numpy, clipped to [0,1]"""
    img = tensor_chw.detach().cpu().float()
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()
    return (img * 255).astype(np.uint8)


def _denorm_vis(vis_batch, device):
    """
    Denormalize CLIP-normalized visible batch back to [0,1].
    vis_batch: (B, 3, H, W)
    """
    mean = _CLIP_MEAN.to(device)
    std  = _CLIP_STD.to(device)
    return (vis_batch * std + mean).clamp(0, 1)


def _norm_ir(ir_batch):
    """
    Normalize infrared to [0,1] for display.
    Per-image min/max normalization.
    ir_batch: (B, 3, H, W) or (B, 1, H, W)
    """
    b = ir_batch.detach().cpu().float()
    out = []
    for i in range(b.shape[0]):
        img = b[i]
        mn, mx = img.min(), img.max()
        img = (img - mn) / (mx - mn + 1e-6)
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        out.append(img)
    return torch.stack(out)   # (B, 3, H, W)


@torch.no_grad()
def visualize_epoch_mask(
    model,
    vis_batch,
    ir_batch,
    epoch,
    save_dir,
    n_samples=4,
    device='cpu'
):
    """
    Visualize the CMDN haze mask for current epoch.

    Parameters:
        model      : Teacher network (VIFNetInconsistencyTeacher from model/Teacher.py)
        vis_batch  : (B, 3, H, W) CLIP-normalized hazy visible image, from real data
        ir_batch   : (B, 3, H, W) infrared image
        epoch      : Current epoch number (for filename)
        save_dir   : Output root directory (usually opt.saved_data_dir)
        n_samples  : Number of samples to show (rows)
        device     : 'cuda' or 'cpu'
    """
    n = min(n_samples, vis_batch.shape[0])
    vis = vis_batch[:n].to(device)
    ir  = ir_batch[:n].to(device)

    training_before = model.training
    model.eval()

    # Use CMDN return_debug to get all intermediate signals
    _model = model.module if hasattr(model, 'module') else model

    try:
        M_vis, disc_pseudo, g_fog, attn_deg, disc, disc_refined = \
            _model.cmdn(vis, ir, disc_alpha=0.0, return_debug=True)
    except Exception as e:
        print(f"[visualize_mask] CMDN forward failed: {e}")
        if training_before:
            model.train()
        return

    if training_before:
        model.train()

    # Move to CPU
    def to_np(t):
        return t.detach().cpu()[:n]

    M_vis = to_np(M_vis)             # (n, 1, H, W)
    g_fog = to_np(g_fog)
    attn_deg = to_np(attn_deg)
    disc_refined = to_np(disc_refined)

    # Binary mask: fixed threshold 0.5
    haze_mask_hard = (M_vis >= 0.5).float()

    # Denormalize visible
    vis_01   = _denorm_vis(vis, device).cpu()   # (n, 3, H, W)
    ir_01    = _norm_ir(ir)                      # (n, 3, H, W)

    # ---- Plotting: 7 columns ----
    fig, axes = plt.subplots(
        nrows=n, ncols=7,
        figsize=(24, 4 * n),
        squeeze=False
    )
    col_titles = [
        'Hazy Visible', 'Infrared',
        'g_fog (CLIP sliding)', 'attn_deg (DINOv2)',
        'disc_refined', 'M_vis (CMDN)', 'Binary Mask (>=0.5)'
    ]

    for row in range(n):
        # Col1: Visible
        axes[row, 0].imshow(_to_numpy_rgb(vis_01[row]))
        axes[row, 0].axis('off')

        # Col2: Infrared
        axes[row, 1].imshow(_to_numpy_rgb(ir_01[row]))
        axes[row, 1].axis('off')

        # Col3: g_fog
        m_gf = g_fog[row, 0].numpy()
        im3 = axes[row, 2].imshow(m_gf, cmap='hot', vmin=0, vmax=1)
        axes[row, 2].axis('off')
        plt.colorbar(im3, ax=axes[row, 2], fraction=0.046, pad=0.04)

        # Col4: attn_deg
        m_ad = attn_deg[row, 0].numpy()
        im4 = axes[row, 3].imshow(m_ad, cmap='hot', vmin=0, vmax=1)
        axes[row, 3].axis('off')
        plt.colorbar(im4, ax=axes[row, 3], fraction=0.046, pad=0.04)

        # Col5: disc_refined
        m_dr = disc_refined[row, 0].numpy()
        im5 = axes[row, 4].imshow(m_dr, cmap='hot', vmin=0, vmax=1)
        axes[row, 4].axis('off')
        plt.colorbar(im5, ax=axes[row, 4], fraction=0.046, pad=0.04)

        # Col6: M_vis (soft density)
        m_vis = M_vis[row, 0].numpy()
        im6 = axes[row, 5].imshow(m_vis, cmap='hot', vmin=0, vmax=1)
        axes[row, 5].axis('off')
        plt.colorbar(im6, ax=axes[row, 5], fraction=0.046, pad=0.04)

        # Col7: Binary mask
        m_hard = haze_mask_hard[row, 0].numpy()
        axes[row, 6].imshow(m_hard, cmap='gray', vmin=0, vmax=1)
        axes[row, 6].axis('off')

        axes[row, 0].set_ylabel(f'Sample {row+1}', fontsize=10)

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight='bold')

    fig.suptitle(f'Epoch {epoch} — CMDN Mask Visualization', fontsize=13, y=1.01)
    plt.tight_layout()

    out_dir = os.path.join(save_dir, 'mask_vis')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'epoch_{epoch:03d}.png')
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

    print(f"\n[mask_vis] Epoch {epoch} mask saved -> {out_path}")
