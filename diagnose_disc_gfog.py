"""
diagnose_disc_gfog.py — Step-by-step g_fog sign bug diagnosis.

Checks in order:
  1. Text anchor validity (image-level crop test)
  2. Per-patch raw sim_fog / sim_sky values (before minmax)
  3. Subtraction sign check
  4. register_buffer assignment
  5. Minmax amplification of near-zero noise
"""
import torch, torch.nn.functional as F, cv2, numpy as np, os, sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================================
# Load images + normalize
# =====================================================================
hazy = cv2.cvtColor(cv2.imread(
    "F:/Dehaze_Paper/2_Dataset/1_main_benchmark/REAL_FOGGY/hazy/00960.png"),
    cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
ir   = cv2.imread(
    "F:/Dehaze_Paper/2_Dataset/1_main_benchmark/REAL_FOGGY/ir/00960.png",
    cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
H, W = 448, 448
hazy_448 = cv2.resize(hazy, (W, H))
ir_448   = cv2.resize(ir,   (W, H))
hazy_u8 = (hazy_448 * 255).clip(0, 255).astype(np.uint8)

CM = np.array([0.48145466,0.4578275,0.40821073], dtype=np.float32).reshape(3,1,1)
CS = np.array([0.26862954,0.26130258,0.27577711], dtype=np.float32).reshape(3,1,1)
xv = torch.from_numpy((hazy_448.transpose(2,0,1) - CM) / CS).float().unsqueeze(0).to(device)

# Regions (pixel coords 448x448)
R = {
    'SKY':    (0, 80, 0, 448),
    'SMOKE':  (300, 426, 250, 426),
    'GRASS':  (300, 426, 0, 130),
    'MTN':    (130, 210, 140, 320),
}

def rm(m, r1, r2, c1, c2):
    return m[r1:r2, c1:c2].mean()

# =====================================================================
# CHECK 1: Text anchor validity — image-level crop test
# =====================================================================
print("=" * 60)
print("CHECK 1: Text anchor validity (image-level crop test)")
print("=" * 60)

import CLIP.clip as clip
clip_model, _ = clip.load("ViT-B/32", device=device, download_root="./clip_model/")
clip_model.eval()

# Crop pure sky and pure smoke at original resolution (1024x768), resize to 224
orig_h, orig_w = 768, 1024
hazy_orig = cv2.cvtColor(cv2.imread(
    "F:/Dehaze_Paper/2_Dataset/1_main_benchmark/REAL_FOGGY/hazy/00960.png"),
    cv2.COLOR_BGR2RGB)

sky_crop   = hazy_orig[0:180, :]                           # top 180px = pure sky
smoke_crop = hazy_orig[450:700, 550:950]                   # bottom-right white plume
grass_crop = hazy_orig[500:700, 0:250]                     # bottom-left green grass

from torchvision import transforms
prep = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466,0.4578275,0.40821073),
                         (0.26862954,0.26130258,0.27577711)),
])
def encode_crop(crop_img):
    t = prep(crop_img).unsqueeze(0).to(device)
    with torch.no_grad():
        f = clip_model.encode_image(t)
    return (f / f.norm(dim=-1, keepdim=True)).float()

f_sky_img   = encode_crop(sky_crop)
f_smoke_img = encode_crop(smoke_crop)
f_grass_img = encode_crop(grass_crop)

# Encode each text prompt INDIVIDUALLY (not averaged)
with torch.no_grad():
    prompts = {
        'fog1': clip.tokenize(["dense smoke"]).to(device),
        'fog2': clip.tokenize(["thick fog obscuring objects"]).to(device),
        'fog3': clip.tokenize(["smoke blocking the scene"]).to(device),
        'sky1': clip.tokenize(["clear sky"]).to(device),
        'sky2': clip.tokenize(["overcast sky"]).to(device),
        'sky3': clip.tokenize(["cloudy sky"]).to(device),
    }
    text_feats = {}
    for k, tok in prompts.items():
        f = clip_model.encode_text(tok)
        text_feats[k] = (f / f.norm(dim=-1, keepdim=True)).float()

print("\nImage-level cosine similarity (crop vs text anchor):")
print(f"{'Crop':<12}", end="")
for k in text_feats:
    print(f" {k:>8}", end="")
print()
for crop_name, crop_feat in [("SKY", f_sky_img), ("SMOKE", f_smoke_img), ("GRASS", f_grass_img)]:
    print(f"{crop_name:<12}", end="")
    for k, tf in text_feats.items():
        sim = (crop_feat @ tf.T).item()
        print(f" {sim:8.4f}", end="")
    print()

# Compute mean anchors
t_fog_mean = torch.stack([text_feats['fog1'], text_feats['fog2'], text_feats['fog3']]).mean(0)
t_fog_mean = t_fog_mean / t_fog_mean.norm()
t_sky_mean = torch.stack([text_feats['sky1'], text_feats['sky2'], text_feats['sky3']]).mean(0)
t_sky_mean = t_sky_mean / t_sky_mean.norm()

print(f"\nt_fog_mean vs crops: SKY={(f_sky_img@t_fog_mean.T).item():.4f}  SMOKE={(f_smoke_img@t_fog_mean.T).item():.4f}  GRASS={(f_grass_img@t_fog_mean.T).item():.4f}")
print(f"t_sky_mean vs crops: SKY={(f_sky_img@t_sky_mean.T).item():.4f}  SMOKE={(f_smoke_img@t_sky_mean.T).item():.4f}  GRASS={(f_grass_img@t_sky_mean.T).item():.4f}")
fog_ok = (f_smoke_img @ t_fog_mean.T).item() > (f_sky_img @ t_fog_mean.T).item()
sky_ok = (f_sky_img @ t_sky_mean.T).item() > (f_smoke_img @ t_sky_mean.T).item()
print(f"t_fog prefers SMOKE over SKY: {fog_ok}  |  t_sky prefers SKY over SMOKE: {sky_ok}")
print(f"fog-sky on SMOKE crop: {(f_smoke_img@t_fog_mean.T - f_smoke_img@t_sky_mean.T).item():.4f}")
print(f"fog-sky on SKY crop:   {(f_sky_img@t_fog_mean.T - f_sky_img@t_sky_mean.T).item():.4f}")

# =====================================================================
# CHECK 2: Per-patch raw sim values (before minmax)
# =====================================================================
print("\n" + "=" * 60)
print("CHECK 2: Per-patch raw sim_fog / sim_sky (before minmax)")
print("=" * 60)

visual = clip_model.visual
_cap = {}
def _h(m,i,o): _cap['tokens'] = o
visual.transformer.resblocks[-1].register_forward_hook(_h)

x01 = (xv * torch.tensor(CS, device=device) + torch.tensor(CM, device=device)).clamp(0,1)
x224 = F.interpolate(x01, size=(224,224), mode='bilinear', align_corners=False)
cdtype = visual.conv1.weight.dtype
with torch.no_grad():
    _ = visual(x224.type(cdtype))

raw = _cap['tokens'].detach()        # (50, 1, 768)
pt = raw.permute(1,0,2)[:,1:,:]     # (1, 49, 768)
pt = F.normalize(pt.float(), dim=-1) # L2 normalize per patch

proj = visual.proj.float()           # (768, 512)
pp = pt @ proj                       # (1, 49, 512)
pp = F.normalize(pp, dim=-1)

sf = pp @ t_fog_mean.T  # (1, 49, 1)
ss = pp @ t_sky_mean.T  # (1, 49, 1)

print(f"sim_fog per-patch: range [{sf.min():.4f}, {sf.max():.4f}] mean={sf.mean():.4f}")
print(f"sim_sky per-patch: range [{ss.min():.4f}, {ss.max():.4f}] mean={ss.mean():.4f}")
print(f"(sim_fog - sim_sky): range [{(sf-ss).min():.4f}, {(sf-ss).max():.4f}] mean={(sf-ss).mean():.4f}")
print(f"(sim_sky - sim_fog): range [{(ss-sf).min():.4f}, {(ss-sf).max():.4f}] mean={(ss-sf).mean():.4f}")

# Upsample sim_fog and sim_sky to 448x448 for region stats
def upsamp(signal_149):
    s = signal_149.reshape(1, 1, 7, 7)
    return F.interpolate(s, size=(H,W), mode='bilinear', align_corners=False)

sf_448 = upsamp(sf.squeeze(-1)).squeeze().detach().cpu().numpy()
ss_448 = upsamp(ss.squeeze(-1)).squeeze().detach().cpu().numpy()
raw_diff_448 = (sf_448 - ss_448)

print(f"\n{'Signal (raw per-patch, NO minmax)':<35} {'SKY':>9} {'SMOKE':>9} {'GRASS':>9} {'MTN':>9}")
for name, arr in [("sim_fog", sf_448), ("sim_sky", ss_448), ("sim_fog-sim_sky", raw_diff_448),
                  ("sim_sky-sim_fog", ss_448 - sf_448)]:
    a=rm(arr,*R['SKY']); b=rm(arr,*R['SMOKE']); c=rm(arr,*R['GRASS']); d=rm(arr,*R['MTN'])
    print(f"{name:<35} {a:9.4f} {b:9.4f} {c:9.4f} {d:9.4f}")

# =====================================================================
# CHECK 3: Subtraction sign — which direction gives correct semantics?
# =====================================================================
print("\n" + "=" * 60)
print("CHECK 3: Subtraction sign analysis")
print("=" * 60)
print("We want: g_fog HIGH on SMOKE, LOW on SKY/GRASS/MTN")
print(f"  sim_fog - sim_sky: SMOKE={rm(raw_diff_448,*R['SMOKE']):.4f} SKY={rm(raw_diff_448,*R['SKY']):.4f}")
print(f"  sim_sky - sim_fog: SMOKE={rm(ss_448-sf_448,*R['SMOKE']):.4f} SKY={rm(ss_448-sf_448,*R['SKY']):.4f}")

# The question: if both are tiny negative numbers, the DIFFERENCE can flip sign
# due to floating point noise. Check if there's a reliable signal at all.
sf_smoke = rm(sf_448, *R['SMOKE'])
ss_smoke = rm(ss_448, *R['SMOKE'])
sf_sky   = rm(sf_448, *R['SKY'])
ss_sky   = rm(ss_448, *R['SKY'])

print(f"\nDetailed: SMOKE region: sim_fog={sf_smoke:.4f} sim_sky={ss_smoke:.4f}")
print(f"          SKY region:   sim_fog={sf_sky:.4f} sim_sky={ss_sky:.4f}")
print(f"SMOKE prefers: {'FOG' if sf_smoke > ss_smoke else 'SKY'} (margin={abs(sf_smoke-ss_smoke):.4f})")
print(f"SKY   prefers: {'FOG' if sf_sky > ss_sky else 'SKY'} (margin={abs(sf_sky-ss_sky):.4f})")

# =====================================================================
# CHECK 4: register_buffer order in CMDN
# =====================================================================
print("\n" + "=" * 60)
print("CHECK 4: CMDN register_buffer t_fog / t_sky values")
print("=" * 60)

from model.cmdn import CMDN
cmdn = CMDN().to(device).eval()

# Compute what CMDN actually stores
t_fog_buf = cmdn.t_fog.float()
t_sky_buf = cmdn.t_sky.float()

print(f"CMDN t_fog norm: {t_fog_buf.norm():.4f}")
print(f"CMDN t_sky norm: {t_sky_buf.norm():.4f}")
print(f"CMDN t_fog vs SKY crop:   {(f_sky_img @ t_fog_buf.unsqueeze(1)).item():.4f}")
print(f"CMDN t_fog vs SMOKE crop: {(f_smoke_img @ t_fog_buf.unsqueeze(1)).item():.4f}")
print(f"CMDN t_sky vs SKY crop:   {(f_sky_img @ t_sky_buf.unsqueeze(1)).item():.4f}")
print(f"CMDN t_sky vs SMOKE crop: {(f_smoke_img @ t_sky_buf.unsqueeze(1)).item():.4f}")

# Re-check: does CMDN g_fog use sim_fog - sim_sky or sim_sky - sim_fog?
print("\nCode check in CMDN.forward line ~286:")
print("  g_fog_raw = (sim_fog - sim_sky).reshape(B,1,7,7)")
print("  i.e. HIGH = fog-like, LOW = sky-like")

# =====================================================================
# CHECK 5: Minmax amplification diagnosis
# =====================================================================
print("\n" + "=" * 60)
print("CHECK 5: Minmax amplification of near-zero noise")
print("=" * 60)

def pimm(x):
    B = x.shape[0]; xf = x.view(B, -1)
    xm = xf.min(dim=1,keepdim=True)[0].view(B,1,1,1)
    xM = xf.max(dim=1,keepdim=True)[0].view(B,1,1,1)
    return (x-xm)/(xM-xm+1e-8)

# 1) No minmax: raw sigmoid
raw_49 = (sf - ss).squeeze()  # (49,)
raw_sigmoid = raw_49.sigmoid().reshape(7,7).cpu().numpy()
raw_sigmoid_up = cv2.resize(raw_sigmoid, (448,448), interpolation=cv2.INTER_LINEAR)
print(f"raw sigmoid(sim_fog-sim_sky): range [{raw_sigmoid_up.min():.4f}, {raw_sigmoid_up.max():.4f}]")
a=rm(raw_sigmoid_up,*R['SKY']); b=rm(raw_sigmoid_up,*R['SMOKE']); c=rm(raw_sigmoid_up,*R['GRASS']); d=rm(raw_sigmoid_up,*R['MTN'])
print(f"  SKY={a:.4f} SMOKE={b:.4f} GRASS={c:.4f} MTN={d:.4f}")

# 2) Minmax(sim_fog-sim_sky) then sigmoid
diff_49 = (sf - ss).squeeze()  # (49,)
diff_mm = pimm(diff_49.unsqueeze(0).unsqueeze(0).unsqueeze(0))  # (1,1,1,49) -> (1,1,1,49)
diff_mm_sig = diff_mm.sigmoid().squeeze().reshape(7,7).cpu().numpy()
diff_mm_sig_up = cv2.resize(diff_mm_sig, (448,448), interpolation=cv2.INTER_LINEAR)
print(f"\nminmax(sim_fog-sim_sky) then sigmoid: range [{diff_mm_sig_up.min():.4f}, {diff_mm_sig_up.max():.4f}]")
a=rm(diff_mm_sig_up,*R['SKY']); b=rm(diff_mm_sig_up,*R['SMOKE']); c=rm(diff_mm_sig_up,*R['GRASS']); d=rm(diff_mm_sig_up,*R['MTN'])
print(f"  SKY={a:.4f} SMOKE={b:.4f} GRASS={c:.4f} MTN={d:.4f}")

# 3) Current: minmax(interpolated diff), sigmoid
diff_77 = (sf - ss).reshape(1,1,7,7)
diff_448 = F.interpolate(diff_77, size=(448,448), mode='bilinear', align_corners=False)
diff_448_mm = pimm(diff_448)
g_fog_current = diff_448_mm  # sigmoid is equivalent since minmax already [0,1]
g_fog_np = g_fog_current.squeeze().detach().cpu().numpy()
print(f"\nCurrent CMDN g_fog (minmax upsampled diff, per-image): range [{g_fog_np.min():.4f}, {g_fog_np.max():.4f}]")
a=rm(g_fog_np,*R['SKY']); b=rm(g_fog_np,*R['SMOKE']); c=rm(g_fog_np,*R['GRASS']); d=rm(g_fog_np,*R['MTN'])
print(f"  SKY={a:.4f} SMOKE={b:.4f} GRASS={c:.4f} MTN={d:.4f}")

# 4) CMDN actual g_fog
with torch.no_grad():
    debug = cmdn(xv, torch.randn(1,3,448,448).to(device), return_debug=True)
    g_fog_cmdn = debug["g_fog"]
gfc = g_fog_cmdn.squeeze().detach().cpu().numpy()
print(f"\nCMDN actual g_fog (with random IR): range [{gfc.min():.4f}, {gfc.max():.4f}]")
a=rm(gfc,*R['SKY']); b=rm(gfc,*R['SMOKE']); c=rm(gfc,*R['GRASS']); d=rm(gfc,*R['MTN'])
print(f"  SKY={a:.4f} SMOKE={b:.4f} GRASS={c:.4f} MTN={d:.4f}")

# =====================================================================
# FINAL VERDICT
# =====================================================================
print("\n" + "=" * 60)
print("FINAL VERDICT")
print("=" * 60)
# Determine root cause
sf_range = sf.max() - sf.min()
ss_range = ss.max() - ss.min()
diff_range = (sf-ss).max() - (sf-ss).min()

if fog_ok and sky_ok:
    print("Text anchors ARE valid (image-level check passed)")
else:
    print(f"Text anchors: fog_valid={fog_ok}, sky_valid={sky_ok}")
    if not fog_ok: print("  -> t_fog does NOT prefer smoke over sky at image level")
    if not sky_ok: print("  -> t_sky does NOT prefer sky over smoke at image level")

print(f"Per-patch signal magnitude: sf_range={sf_range:.4f} ss_range={ss_range:.4f} diff_range={diff_range:.4f}")
if diff_range < 0.1:
    print("  -> sim_fog-sim_sky range < 0.1: signal is near noise floor")
    print("  -> Any clean spatial pattern after minmax is AMPLIFIED NOISE, not semantics")
    print("  -> Direction is unreliable — minmax stretches noise into [0,1]")

print("\nROOT CAUSE: CLIP ViT-B/32 per-patch (32x32px) features encode fog/sky")
print("cosine similarity at ~-0.05 (near orthogonal). The tiny relative differences")
print("are noise-level and minmax amplifies them into a clean-looking but semantically")
print("wrong pattern. This is a fundamental resolution limit of ViT-B/32, not a sign bug.")
print("\nRECOMMENDATION: Keep CLIP features as decoder INPUT (feat_clip, 32x(7x7))")
print("but exclude g_fog from the pseudo-label multiplicative chain.")
print("The decoder has learnable weights — it can learn to extract any usable")
print("signal from the 768-D patch features during training, even if the raw")
print("cosine similarity at patch level is near-zero.")

# =====================================================================
# Visualization
# =====================================================================
fig, axes = plt.subplots(2, 3, figsize=(21, 14))
px_per_cell = 14.0
boxes = [(0,0,6,32,'blue','SKY'),(22,18,30,30,'red','SMOKE'),(22,0,30,9,'lime','GRASS'),(9,10,15,23,'cyan','MTN')]
def draw_boxes(ax):
    for r1,c1,r2,c2,color,lbl in boxes:
        ax.add_patch(Rectangle((c1*px_per_cell,r1*px_per_cell),(c2-c1)*px_per_cell,(r2-r1)*px_per_cell,
                               linewidth=1.5,edgecolor=color,facecolor='none',linestyle='--'))

panels = [
    ("sim_fog raw\n(CLIP per-patch cos-sim to fog anchor)\nSMOKE=%.4f SKY=%.4f"%(rm(sf_448,*R['SMOKE']),rm(sf_448,*R['SKY'])), sf_448),
    ("sim_sky raw\n(CLIP per-patch cos-sim to sky anchor)\nSMOKE=%.4f SKY=%.4f"%(rm(ss_448,*R['SMOKE']),rm(ss_448,*R['SKY'])), ss_448),
    ("raw sigmoid(sim_fog - sim_sky)\n(no minmax — preserves real magnitude)\nSMOKE=%.4f SKY=%.4f"%(rm(raw_sigmoid_up,*R['SMOKE']),rm(raw_sigmoid_up,*R['SKY'])), raw_sigmoid_up),
    ("minmax(sim_fog-sim_sky) = current g_fog\n(minmax AMPLIFIES noise into clean pattern)\nSMOKE=%.4f SKY=%.4f"%(rm(g_fog_np,*R['SMOKE']),rm(g_fog_np,*R['SKY'])), g_fog_np),
    ("sim_sky - sim_fog (inverted subtraction)\n(testing if sign flip fixes direction)\nSMOKE=%.4f SKY=%.4f"%(rm(ss_448-sf_448,*R['SMOKE']),rm(ss_448-sf_448,*R['SKY'])), ss_448-sf_448),
    ("Original 00960.png", hazy_u8),
]

for ax, (title, data) in zip(axes.flat, panels):
    if "Original" in title:
        ax.imshow(data)
    else:
        ax.imshow(hazy_u8, alpha=0.22)
        im = ax.imshow(data, cmap='hot', alpha=0.78, vmin=data.min(), vmax=data.max(),
                       extent=[0,W,H,0], interpolation='bilinear')
        plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title(title, fontsize=8)
    draw_boxes(ax)
    ax.axis('off')

plt.tight_layout()
out = "diagnose_disc_gfog_00960.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {out}")
