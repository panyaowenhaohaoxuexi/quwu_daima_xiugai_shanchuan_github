"""
verify_cmdn_semantics.py — Final CMDN semantic direction check.

Loads 00960 (hazy + IR), runs CMDN with return_debug=True,
prints per-signal region statistics and saves a 6-panel overlay PNG.
"""
import torch, cv2, numpy as np
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

parser = argparse.ArgumentParser()
parser.add_argument('--hazy', type=str,
                    default='F:/Dehaze_Paper/2_Dataset/1_main_benchmark/REAL_FOGGY/hazy/00887.png')
parser.add_argument('--ir',   type=str,
                    default='F:/Dehaze_Paper/2_Dataset/1_main_benchmark/REAL_FOGGY/ir/00887.png')
parser.add_argument('--out',  type=str, default='cmdn_verify_results/cmdn_semantic_check_00887.png')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ---------------------------------------------------------------------------
# Load & normalise
# ---------------------------------------------------------------------------
hazy = cv2.cvtColor(cv2.imread(args.hazy), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
ir   = cv2.imread(args.ir, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
H, W = 448, 448
hazy = cv2.resize(hazy, (W, H))
ir   = cv2.resize(ir,   (W, H))

CM = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32).reshape(3, 1, 1)
CS = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32).reshape(3, 1, 1)
xv = torch.from_numpy((hazy.transpose(2, 0, 1) - CM) / CS).float().unsqueeze(0).to(device)
xi = torch.from_numpy(np.stack([ir] * 3, axis=0)).float().unsqueeze(0).to(device)
hazy_u8 = (hazy * 255).clip(0, 255).astype(np.uint8)

# ---------------------------------------------------------------------------
# CMDN forward
# ---------------------------------------------------------------------------
from model.cmdn import CMDN

cmdn = CMDN().to(device).eval()
with torch.no_grad():
    debug = cmdn(xv, xi, return_debug=True)

def to_np(t):
    return t.squeeze().cpu().numpy()

maps = dict(
    g_fog=to_np(debug["g_fog"]),
    attn_deg=to_np(debug["attn_deg"]),
    disc_refined=to_np(debug["disc_refined"]),
    P_fail=to_np(debug["P_fail"]),
    P_pseudo=to_np(debug["P_pseudo"]),
    G_dec=to_np(debug["G_dec"]),
    P_support=to_np(debug["P_support"]),
    G_soft=to_np(debug["G_soft"]),
    M_hard=to_np(debug["M_hard"]),
)

# ---------------------------------------------------------------------------
# Region statistics (pixel coords on 448x448, plus 32x32 grid core patches)
# ---------------------------------------------------------------------------
# Pixel-level bounding boxes
regions_px = {
    'A:SMOKE (px)':     (266, 426, 224, 426),
    'B:GRASS (px)':     (266, 426, 0, 157),
    'C:MOUNTAIN (px)':  (112, 224, 112, 336),
}

# 32x32 core patches (deep inside each region, avoiding boundaries)
# Convert to 448x448 for direct comparison
def grid_to_px(r1, r2, c1, c2):
    return (r1*14, r2*14, c1*14, c2*14)

regions_core = {
    'A:SMOKE (core)':   grid_to_px(22, 28, 18, 24),  # deep inside white smoke
    'B:GRASS (core)':   grid_to_px(22, 28, 2, 8),    # deep inside green grass
    'C:MTN (core)':     grid_to_px(10, 14, 10, 16),  # central mountain
}

def region_mean(m, r1, r2, c1, c2):
    return m[r1:r2, c1:c2].mean()

print("\n--- Region means (pixel boxes) ---")
hdr = f"{'Metric':<18} {'A:SMOKE':>9} {'B:GRASS':>9} {'C:MTN':>9}"
print(hdr)
print("-" * len(hdr))
for name, m in maps.items():
    a = region_mean(m, *regions_px['A:SMOKE (px)'])
    b = region_mean(m, *regions_px['B:GRASS (px)'])
    c = region_mean(m, *regions_px['C:MOUNTAIN (px)'])
    print(f"{name:<18} {a:9.4f} {b:9.4f} {c:9.4f}")

print("\n--- Core patch means (32x32, boundary-avoiding) ---")
for name, m in maps.items():
    a = region_mean(m, *regions_core['A:SMOKE (core)'])
    b = region_mean(m, *regions_core['B:GRASS (core)'])
    c = region_mean(m, *regions_core['C:MTN (core)'])
    print(f"{name:<18} {a:9.4f} {b:9.4f} {c:9.4f}")

# Print P95/P5 for smoke-core vs grass-core
dr = maps['disc_refined']
sm_core = dr[regions_core['A:SMOKE (core)'][0]:regions_core['A:SMOKE (core)'][1],
             regions_core['A:SMOKE (core)'][2]:regions_core['A:SMOKE (core)'][3]].flatten()
gr_core = dr[regions_core['B:GRASS (core)'][0]:regions_core['B:GRASS (core)'][1],
             regions_core['B:GRASS (core)'][2]:regions_core['B:GRASS (core)'][3]].flatten()
p95s = np.percentile(sm_core, 95)
p5g  = np.percentile(gr_core, 5)
print(f"\ndisc_refined smoke P95={p95s:.4f}  grass P5={p5g:.4f}  P95-P5 gap={p95s-p5g:.3f}")

# Quick stats
for name, m in maps.items():
    print(f"  {name:15s}: range [{m.min():.4f}, {m.max():.4f}], mean={m.mean():.4f}")

# ---------------------------------------------------------------------------
# 6-panel PNG
# ---------------------------------------------------------------------------
boxes_grid = [
    (17, 16, 30, 30, 'red',  'A:SMOKE'),
    (19, 0,  30, 11, 'lime', 'B:GRASS'),
    (8,  8,  16, 24, 'cyan', 'C:MTN'),
]
px_per_cell = 14.0

def draw_boxes(ax):
    for r1, c1, r2, c2, color, lbl in boxes_grid:
        rect = Rectangle((c1*px_per_cell, r1*px_per_cell),
                         (c2-c1)*px_per_cell, (r2-r1)*px_per_cell,
                         linewidth=1.5, edgecolor=color, facecolor='none', linestyle='--')
        ax.add_patch(rect)

fig, axes = plt.subplots(2, 3, figsize=(21, 14))

titles = [
    ("g_fog\nCLIP fog-sky diff (per-image minmax)\nNOT in pseudo-label (direction unreliable)", maps["g_fog"]),
    ("attn_deg\nDINOv2 x_prenorm 3x3 local structure variance\nHIGH=textured/clear, LOW=uniform/smoke", maps["attn_deg"]),
    ("disc_refined = pseudo-label source\nSmoke core P95={:.3f}, Grass core P5={:.3f}".format(p95s, p5g), maps["disc_refined"]),
    ("P_fail = decoder output\nUntrained -- check [0,1] range only", maps["P_fail"]),
    ("G_soft = G_dec * P_support\nFinal conservative soft gate", maps["G_soft"]),
    ("Original 00960.png (448x448)", np.ones((448, 448), dtype=np.float32)),
]

for ax, (title, data) in zip(axes.flat, titles):
    if "Original" in title:
        ax.imshow(hazy_u8)
        ax.set_title(title, fontsize=9)
        draw_boxes(ax)
        ax.axis('off')
    else:
        m = data if isinstance(data, np.ndarray) else to_np(data)
        ax.imshow(hazy_u8, alpha=0.25)
        im = ax.imshow(m, cmap='hot', alpha=0.75, vmin=0, vmax=1,
                       extent=[0, W, H, 0], interpolation='bilinear')
        ax.set_title(title, fontsize=8)
        draw_boxes(ax)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
out = args.out
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"\nSaved: {out}")
