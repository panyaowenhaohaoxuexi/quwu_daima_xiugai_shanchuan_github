"""
verify_cmdn_fusion.py — Three-way fusion test on 00960 with alpha sweep.
Tests: alpha=0 (cold start), alpha=0.5, alpha=1.0 (full disc).
"""
import torch, cv2, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load 00960
hazy = cv2.cvtColor(cv2.imread(
    "F:/Dehaze_Paper/2_Dataset/1_main_benchmark/REAL_FOGGY/hazy/00960.png"),
    cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
ir   = cv2.imread(
    "F:/Dehaze_Paper/2_Dataset/1_main_benchmark/REAL_FOGGY/ir/00960.png",
    cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
H, W = 448, 448
hazy_448 = cv2.resize(hazy, (W, H)); ir_448 = cv2.resize(ir, (W, H))
ir_448 = np.stack([ir_448]*3, axis=0)
hazy_u8 = (hazy_448 * 255).clip(0, 255).astype(np.uint8)

CM = np.array([0.48145466,0.4578275,0.40821073], dtype=np.float32).reshape(3,1,1)
CS = np.array([0.26862954,0.26130258,0.27577711], dtype=np.float32).reshape(3,1,1)
xv = torch.from_numpy((hazy_448.transpose(2,0,1) - CM) / CS).float().unsqueeze(0).to(device)
xi = torch.from_numpy(ir_448).float().unsqueeze(0).to(device)

# Regions
R = {'SKY':(0,80,0,448), 'SMOKE':(300,426,250,426), 'GRASS':(300,426,0,130), 'MTN':(130,210,140,320)}
def rm(m,r1,r2,c1,c2): return m[r1:r2,c1:c2].mean()

from model.cmdn import CMDN

print("=" * 70)
print("THREE-WAY FUSION — alpha sweep on 00960")
print("=" * 70)

box_data = []
for alpha in [0.0, 0.5, 1.0]:
    cmdn = CMDN().to(device).eval()
    with torch.no_grad():
        M_vis, disc_pseudo, g_fog, attn_deg, disc, disc_refined = \
            cmdn(xv, xi, disc_alpha=alpha, return_debug=True)

    def tonp(t): return t.squeeze().detach().cpu().numpy()
    gf = tonp(g_fog); ad = tonp(attn_deg); dc = tonp(disc); dr = tonp(disc_refined)

    print(f"\nalpha={alpha:.1f}:")
    print(f"  {'Signal':<20} {'SKY':>8} {'SMOKE':>8} {'GRASS':>8} {'MTN':>8} {'SM-SK':>8}")
    for name, m in [("g_fog",gf), ("attn_deg",ad), ("disc",dc), ("disc_refined",dr)]:
        a=rm(m,*R['SKY']); b=rm(m,*R['SMOKE']); c=rm(m,*R['GRASS']); d=rm(m,*R['MTN'])
        print(f"  {name:<20} {a:8.4f} {b:8.4f} {c:8.4f} {d:8.4f} {b-a:+8.4f}")
    print(f"  disc_refined range: [{dr.min():.4f}, {dr.max():.4f}] mean={dr.mean():.4f}")
    box_data.append((alpha, gf, ad, dc, dr))

# =====================================================================
# 6-panel figure: alpha=0 (cold start) — full view
# =====================================================================
alpha0_gf, alpha0_ad, alpha0_dc, alpha0_dr = box_data[0][1:]
alpha05_dr = box_data[1][4]
alpha10_dr = box_data[2][4]

fig, axes = plt.subplots(2, 3, figsize=(22, 14))
px_per_cell = 448/32
boxes = [(0,0,6,32,'blue','SKY'),(22,18,30,30,'red','SMOKE'),(22,0,30,9,'lime','GRASS'),(9,10,15,23,'cyan','MTN')]
def db(ax):
    for r1,c1,r2,c2,clr,lbl in boxes:
        ax.add_patch(Rectangle((c1*px_per_cell,r1*px_per_cell),(c2-c1)*px_per_cell,(r2-r1)*px_per_cell,
                               linewidth=1.5,edgecolor=clr,facecolor='none',linestyle='--'))

panels = [
    ("g_fog (sliding CLS)\nSMOKE=%.3f SKY=%.3f GRASS=%.3f"%
     (rm(alpha0_gf,*R['SMOKE']),rm(alpha0_gf,*R['SKY']),rm(alpha0_gf,*R['GRASS'])), alpha0_gf),
    ("attn_deg (DINOv2 structure)\nSMOKE=%.3f GRASS=%.3f (high=clear)"%
     (rm(alpha0_ad,*R['SMOKE']),rm(alpha0_ad,*R['GRASS'])), alpha0_ad),
    ("disc (VIS-IR gap, random encoder)\nSMOKE=%.3f SKY=%.3f GRASS=%.3f"%
     (rm(alpha0_dc,*R['SMOKE']),rm(alpha0_dc,*R['SKY']),rm(alpha0_dc,*R['GRASS'])), alpha0_dc),
    ("disc_refined alpha=0 (cold start)\ng_fog * (1-attn_deg)  NO disc\nSM-SK=%.3f SM-GR=%.3f"%
     (rm(alpha0_dr,*R['SMOKE'])-rm(alpha0_dr,*R['SKY']), rm(alpha0_dr,*R['SMOKE'])-rm(alpha0_dr,*R['GRASS'])), alpha0_dr),
    ("disc_refined alpha=0.5 (half disc)\nSM-SK=%.3f SM-GR=%.3f"%
     (rm(alpha05_dr,*R['SMOKE'])-rm(alpha05_dr,*R['SKY']), rm(alpha05_dr,*R['SMOKE'])-rm(alpha05_dr,*R['GRASS'])), alpha05_dr),
    ("disc_refined alpha=1.0 (full disc)\nSM-SK=%.3f SM-GR=%.3f"%
     (rm(alpha10_dr,*R['SMOKE'])-rm(alpha10_dr,*R['SKY']), rm(alpha10_dr,*R['SMOKE'])-rm(alpha10_dr,*R['GRASS'])), alpha10_dr),
]

for ax, (title, data) in zip(axes.flat, panels):
    ax.imshow(hazy_u8, alpha=0.22)
    im = ax.imshow(data, cmap='hot', alpha=0.78, vmin=0, vmax=1,
                   extent=[0,W,H,0], interpolation='bilinear')
    ax.set_title(title, fontsize=8.5)
    db(ax); ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046)

# Verdict
sm_sk0 = rm(alpha0_dr,*R['SMOKE']) - rm(alpha0_dr,*R['SKY'])
sm_gr0 = rm(alpha0_dr,*R['SMOKE']) - rm(alpha0_dr,*R['GRASS'])
ok = sm_sk0 > 0.3 and sm_gr0 > 0.15
fig.suptitle(f"CMDN Three-Way Fusion — alpha sweep (gamma=1.5)\n"
             f"Cold start (alpha=0): SMOKE-SKY={sm_sk0:+.3f} SMOKE-GRASS={sm_gr0:+.3f}  ->  {'PASS' if ok else 'CHECK'}",
             fontsize=12, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0,0,1,0.95])
out = "cmdn_fusion_sweep_00960.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out}")
print(f"\nCold start verdict: SMOKE-SKY={sm_sk0:+.3f} SMOKE-GRASS={sm_gr0:+.3f} -> {'PASS' if ok else 'CHECK'}")
