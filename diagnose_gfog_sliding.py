"""
diagnose_gfog_sliding.py — CLS sliding-window g_fog test on 00960.

Grid: 7x7, input 448x448 -> each crop is 224x224 (RESIZE-based, not extract).
Each crop is independently CLIP-normalised and fed to visual().
CLS token (image-level) is used to compute fog-sky similarity.
49 values -> 7x7 grid -> upsample to 448 -> sigmoid -> g_fog.
"""
import torch, cv2, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================================
# Load image
# =====================================================================
hazy = cv2.cvtColor(cv2.imread(
    "F:/Dehaze_Paper/2_Dataset/1_main_benchmark/REAL_FOGGY/hazy/00960.png"),
    cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
H, W = 448, 448
hazy_448 = cv2.resize(hazy, (W, H))
hazy_u8 = (hazy_448 * 255).clip(0, 255).astype(np.uint8)

# =====================================================================
# Load CLIP
# =====================================================================
import CLIP.clip as clip
clip_model, _ = clip.load("ViT-B/32", device=device, download_root="./clip_model/")
clip_model.eval()

# Pre-compute text anchors (frozen)
with torch.no_grad():
    t_fog = clip_model.encode_text(clip.tokenize([
        "dense smoke", "thick fog obscuring objects",
        "smoke blocking the scene",
    ]).to(device)).mean(0).float()  # (512,) fp32
    t_sky = clip_model.encode_text(clip.tokenize([
        "clear sky", "overcast sky", "cloudy sky",
    ]).to(device)).mean(0).float()  # (512,) fp32

# =====================================================================
# Sliding window: 7x7 grid, each cell is a center point
# For each center (cx, cy) in 448 coordinates, crop 224x224 around it.
# Grid spacing = 448/7 = 64 px. Crop half-size = 112 px.
# =====================================================================
GRID = 7
grid_step = H / GRID      # 64.0
crop_half = H / 2         # 224 — full image size equals one crop

# CLIP normalisation transform
CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
CLIP_STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711],  device=device).view(1, 3, 1, 1)

hazy_t = torch.from_numpy(hazy_448.transpose(2, 0, 1)).float().unsqueeze(0).to(device)  # (1,3,448,448)

g_fog_grid = np.zeros((GRID, GRID), dtype=np.float32)
visual = clip_model.visual

with torch.no_grad():
    for i in range(GRID):
        for j in range(GRID):
            # Center of this grid cell in pixel coords
            cy = int(grid_step * i + grid_step / 2)  # 32, 96, 160, ...
            cx = int(grid_step * j + grid_step / 2)

            # Crop bounds (with clamping for edges)
            y1 = max(0, cy - 112)
            y2 = min(448, y1 + 224)
            x1 = max(0, cx - 112)
            x2 = min(448, x1 + 224)
            # Re-adjust y1/x1 if we hit the right/bottom boundary
            y1 = y2 - 224
            x1 = x2 - 224

            crop = hazy_t[:, :, y1:y2, x1:x2]  # (1, 3, 224, 224)

            # Crop is already in [0,1] from the original resized image.
            # But the crop might not be CLIP-normalized. The original full image
            # was resized to 448 and denormalized. Each crop is a subregion.
            # We CLIP-normalize the crop directly (it's in [0,1] pixel space).
            crop_norm = (crop - CLIP_MEAN) / CLIP_STD

            # CLIP expects 224x224, crop already is
            clip_feat = visual(crop_norm.type(visual.conv1.weight.dtype))  # (1, 512) — CLS token
            clip_feat = clip_feat.float()
            clip_feat = clip_feat / clip_feat.norm(dim=-1, keepdim=True)

            sim_fog = (clip_feat @ t_fog.unsqueeze(1)).item()
            sim_sky = (clip_feat @ t_sky.unsqueeze(1)).item()
            g_fog_grid[i, j] = sim_fog - sim_sky

# =====================================================================
# Upsample 7x7 grid -> 448x448, sigmoid
# =====================================================================
g_fog_t = torch.from_numpy(g_fog_grid).float().unsqueeze(0).unsqueeze(0)  # (1,1,7,7)
g_fog_up = torch.nn.functional.interpolate(g_fog_t, size=(H, W), mode='bilinear',
                                            align_corners=False)
g_fog_up_sigmoid = g_fog_up.sigmoid().squeeze().numpy()
g_fog_up_raw = g_fog_up.squeeze().numpy()  # before sigmoid

# Per-image minmax version
g_fog_mm = (g_fog_up - g_fog_up.min()) / (g_fog_up.max() - g_fog_up.min() + 1e-8)
g_fog_mm_np = g_fog_mm.squeeze().numpy()

print("=" * 60)
print("CLS SLIDING-WINDOW g_fog RESULTS")
print("=" * 60)
print(f"7x7 raw grid (sim_fog - sim_sky):")
for i in range(GRID):
    print(" ".join(f"{g_fog_grid[i,j]:+.4f}" for j in range(GRID)))
print(f"\nGrid range: [{g_fog_grid.min():.4f}, {g_fog_grid.max():.4f}]")
print(f"Sigmoid range: [{g_fog_up_sigmoid.min():.4f}, {g_fog_up_sigmoid.max():.4f}]")
print(f"Minmax range: [{g_fog_mm_np.min():.4f}, {g_fog_mm_np.max():.4f}]")

# =====================================================================
# Region stats
# =====================================================================
R = {
    'SKY':    (0, 80, 0, 448),
    'SMOKE':  (300, 426, 250, 426),
    'GRASS':  (300, 426, 0, 130),
    'MTN':    (130, 210, 140, 320),
}
def rm(m, r1,r2,c1,c2): return m[r1:r2, c1:c2].mean()

print(f"\n{'Signal':<25} {'SKY':>9} {'SMOKE':>9} {'GRASS':>9} {'MTN':>9}")
print("-" * 65)

# Show raw grid values at 7x7 level to avoid upsampling blur
g_fog_sig = g_fog_up_sigmoid
g_fog_raw_up = g_fog_up_raw
for name, arr in [
    ("g_fog raw (grid, no sig)", g_fog_grid),
    ("g_fog sigmoid (7x7->448)", g_fog_sig),
    ("g_fog minmax (7x7->448)", g_fog_mm_np),
]:
    if arr.ndim == 2 and arr.shape == (7,7):
        # grid: map to approximate pixel regions
        a=arr[0:2,:].mean(); b=arr[5:7,4:7].mean()
        c=arr[5:7,0:2].mean(); d=arr[2:4,2:5].mean()
        print(f"{name:<25} {a:9.4f} {b:9.4f} {c:9.4f} {d:9.4f}")
    else:
        a=rm(arr,*R['SKY']); b=rm(arr,*R['SMOKE'])
        c=rm(arr,*R['GRASS']); d=rm(arr,*R['MTN'])
        print(f"{name:<25} {a:9.4f} {b:9.4f} {c:9.4f} {d:9.4f}")

# Direction check
margin_sm_gr = g_fog_mm_np[300:426,250:426].mean() - g_fog_mm_np[300:426,0:130].mean()
margin_sm_sk = g_fog_mm_np[300:426,250:426].mean() - g_fog_mm_np[0:80,:].mean()
print(f"\nSMOKE-GRASS margin: {margin_sm_gr:+.4f}")
print(f"SMOKE-SKY margin:   {margin_sm_sk:+.4f}")
direction_ok = margin_sm_gr > 0 and margin_sm_sk > 0
print(f"Direction: {'PASS (SMOKE >> others)' if direction_ok else 'FAIL'}")
if not direction_ok:
    print("!!! g_fog direction is WRONG with sliding-window CLS token")

# =====================================================================
# Compare: per-patch vs sliding-window
# =====================================================================
print("\nCOMPARISON: per-patch (7x7 token) vs sliding-window (7x7 CLS)")
print("per-patch minmax: SKY=0.85 SMOKE=0.30 -> FAIL (direction reversed)")
print(f"sliding-window:   SKY={rm(g_fog_mm_np,*R['SKY']):.2f} SMOKE={rm(g_fog_mm_np,*R['SMOKE']):.2f} -> {'PASS' if direction_ok else 'FAIL'}")

# =====================================================================
# Visualization
# =====================================================================
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
px_per_cell = 64.0
boxes = [(0,0,1.5,7,'blue','SKY'),(5.2,4,7,7,'red','SMOKE'),(5.2,0,7,2,'lime','GRASS'),(2,2.2,4,5.1,'cyan','MTN')]
def draw_boxes(ax):
    for r1,c1,r2,c2,color,lbl in boxes:
        ax.add_patch(Rectangle((c1*px_per_cell,r1*px_per_cell),(c2-c1)*px_per_cell,(r2-r1)*px_per_cell,
                               linewidth=1.5,edgecolor=color,facecolor='none',linestyle='--'))

axes[0].imshow(hazy_u8)
axes[0].set_title("Original 00960.png (448x448)", fontsize=11, fontweight='bold')
draw_boxes(axes[0])
axes[0].axis('off')

axes[1].imshow(hazy_u8, alpha=0.3)
im1 = axes[1].imshow(g_fog_mm_np, cmap='hot', alpha=0.7, vmin=0, vmax=1,
                     extent=[0, W, H, 0], interpolation='bilinear')
axes[1].set_title(f"g_fog = minmax(sliding CLS fog-sky)\n"
                  f"SKY={rm(g_fog_mm_np,*R['SKY']):.3f} SMOKE={rm(g_fog_mm_np,*R['SMOKE']):.3f} "
                  f"GRASS={rm(g_fog_mm_np,*R['GRASS']):.3f} MTN={rm(g_fog_mm_np,*R['MTN']):.3f}\n"
                  f"SMOKE-SKY margin={margin_sm_sk:+.3f} -> {'PASS' if direction_ok else 'FAIL'}",
                  fontsize=10)
draw_boxes(axes[1])
axes[1].axis('off')
plt.colorbar(im1, ax=axes[1], fraction=0.046)

# Also show sigmoid (no minmax) for comparison
axes[2].imshow(hazy_u8, alpha=0.3)
im2 = axes[2].imshow(g_fog_up_sigmoid, cmap='hot', alpha=0.7, vmin=0, vmax=1,
                     extent=[0, W, H, 0], interpolation='bilinear')
axes[2].set_title(f"g_fog sigmoid (no minmax)\n"
                  f"SKY={rm(g_fog_up_sigmoid,*R['SKY']):.3f} SMOKE={rm(g_fog_up_sigmoid,*R['SMOKE']):.3f} "
                  f"GRASS={rm(g_fog_up_sigmoid,*R['GRASS']):.3f} MTN={rm(g_fog_up_sigmoid,*R['MTN']):.3f}",
                  fontsize=10)
draw_boxes(axes[2])
axes[2].axis('off')
plt.colorbar(im2, ax=axes[2], fraction=0.046)

plt.tight_layout()
out = "gfog_sliding_window_00960.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out}")

# =====================================================================
# Performance estimate
# =====================================================================
import time
t0 = time.time()
for _ in range(5):
    for i in range(GRID):
        for j in range(GRID):
            _ = visual(crop_norm.type(visual.conv1.weight.dtype))
t1 = time.time()
ms_per_batch = (t1-t0) / 5 * 1000
print(f"\nPerformance: {ms_per_batch:.0f} ms per batch (49 CLIP forwards, frozen, eval mode)")
print(f"NOTE: Can be reduced by batching all 49 crops into one forward pass (batch_size=49).")
