"""
utils/visualize_mask.py

Haze mask visualization tool during training.
Call visualize_epoch_mask() at the end of each epoch,
saves a 4-column comparison image:
  Col1: Hazy visible image (denormalized to [0,1])
  Col2: Infrared image (normalized to [0,1])
  Col3: HAPM soft density map M_vis (continuous)
  Col4: Binarized haze_mask (Otsu threshold, 0/1)

Saved to {save_dir}/mask_vis/epoch_{epoch:03d}.png each epoch.
Fixed first N samples ensure cross-epoch comparability.
"""

import os
import torch
import torch.nn.functional as F
from model.Teacher import differentiable_otsu
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
    # Per-image independent normalization
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
    Visualize the haze_mask for current epoch.

    Parameters:
        model      : Teacher network (VIFNetInconsistencyTeacher from model/Teacher.py)
        vis_batch  : (B, 3, H, W) CLIP-normalized hazy visible image, from real data
        ir_batch   : (B, 3, H, W) infrared image
        epoch      : Current epoch number (for filename)
        save_dir   : Output root directory (usually opt.saved_data_dir)
        n_samples  : Number of samples to show (rows)
        device     : 'cuda' or 'cpu'
    """
    # Determine actual sample count
    n = min(n_samples, vis_batch.shape[0])
    vis = vis_batch[:n].to(device)
    ir  = ir_batch[:n].to(device)

    # Switch to eval mode, restore afterwards
    training_before = model.training
    model.eval()

    # ---- Hook to capture HDE soft density map M_vis ----
    # HDE outputs M_vis (continuous), which is then binarized by Otsu into haze_mask
    # We capture both soft and hard maps, so hook HDE's output
    _captured = {}

    def _hde_hook(module, inp, out):
        _captured['M_vis'] = out.detach().cpu()   # (B, 1, H, W)

    # Handle DataParallel wrapper
    _model = model.module if hasattr(model, 'module') else model
    _hook = _model.hde.register_forward_hook(_hde_hook)

    try:
        # Normal forward pass (haze_mask is auto-generated internally)
        _ = model(vis, ir)
    except Exception as e:
        print(f"[visualize_mask] Forward inference failed: {e}")
        _hook.remove()
        if training_before:
            model.train()
        return
    finally:
        _hook.remove()

    # Restore training mode
    if training_before:
        model.train()

    if 'M_vis' not in _captured:
        print("[visualize_mask] Failed to capture HDE output, skipping visualization.")
        return

    M_vis = _captured['M_vis'][:n]   # (n, 1, H, W)

    # Recompute haze_mask from M_vis (consistent with model/Teacher.py logic)
    # Use differentiable_otsu for threshold, identical to training forward
    tau = differentiable_otsu(M_vis)                     # (n,1,1,1)
    haze_mask_hard = (M_vis >= tau).float()               # (n,1,H,W) actual training hard mask

    # Denormalize visible
    vis_01   = _denorm_vis(vis, device).cpu()   # (n, 3, H, W)
    ir_01    = _norm_ir(ir)                      # (n, 3, H, W)

    # ---- Plotting ----
    fig, axes = plt.subplots(
        nrows=n, ncols=4,
        figsize=(16, 4 * n),
        squeeze=False
    )
    col_titles = ['Hazy Visible', 'Infrared', 'HAPM Soft Density (M_vis)', 'Binary Haze Mask (Otsu)']

    for row in range(n):
        # Col1: Visible
        axes[row, 0].imshow(_to_numpy_rgb(vis_01[row]))
        axes[row, 0].axis('off')

        # Col2: Infrared
        axes[row, 1].imshow(_to_numpy_rgb(ir_01[row]))
        axes[row, 1].axis('off')

        # Col3: Soft density map (pseudo-color)
        m_soft = M_vis[row, 0].numpy()   # (H, W)
        im3 = axes[row, 2].imshow(m_soft, cmap='hot', vmin=0, vmax=1)
        axes[row, 2].axis('off')
        plt.colorbar(im3, ax=axes[row, 2], fraction=0.046, pad=0.04)

        # Col4: Binary mask
        m_hard = haze_mask_hard[row, 0].numpy()   # (H, W)
        axes[row, 3].imshow(m_hard, cmap='gray', vmin=0, vmax=1)
        axes[row, 3].axis('off')

        # Row label
        axes[row, 0].set_ylabel(f'Sample {row+1}', fontsize=10)

    # Column titles
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=11, fontweight='bold')

    fig.suptitle(f'Epoch {epoch} — Haze Mask Visualization', fontsize=13, y=1.01)
    plt.tight_layout()

    # Save
    out_dir = os.path.join(save_dir, 'mask_vis')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'epoch_{epoch:03d}.png')
    plt.savefig(out_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

    print(f"\n[mask_vis] Epoch {epoch} mask saved -> {out_path}")
