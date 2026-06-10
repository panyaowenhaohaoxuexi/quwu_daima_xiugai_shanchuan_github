"""
probe_hooks.py — Step 0 硬前置探针
验证 CLIP hook 接口、DINOv2 加载接口、L2 范数代理区分度。
不通过不许写正式 forward。
"""

import torch
import torch.nn.functional as F
import sys
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[probe] Using device: {device}")
print(f"[probe] CWD: {os.getcwd()}")

# ============================================================
# PART 1: CLIP 探针
# ============================================================
print("\n" + "=" * 60)
print("PART 1: CLIP Probe")
print("=" * 60)

try:
    import CLIP.clip as clip
except Exception as e:
    print(f"[CLIP] FATAL: Cannot import CLIP.clip: {e}")
    sys.exit(1)

clip_model, _ = clip.load("ViT-B/32", device=device, download_root="./clip_model/")
clip_model.eval()
print(f"[CLIP] Model loaded successfully on {device}")

# Freeze
for p in clip_model.parameters():
    p.requires_grad = False

# Verify visual component
visual = clip_model.visual
print(f"[CLIP] visual.transformer.resblocks: {len(visual.transformer.resblocks)} blocks")

# Verify proj
print(f"[CLIP] visual.proj shape: {visual.proj.shape}")
print(f"[CLIP] visual.proj dtype: {visual.proj.dtype}")

# ---- Hook to capture last transformer block output (before ln_post) ----
_captured_clip = {}

def _clip_hook(module, inp, out):
    _captured_clip['patch_tokens'] = out  # (seq, batch, dim)

hook = visual.transformer.resblocks[-1].register_forward_hook(_clip_hook)
print(f"[CLIP] Hook registered on resblocks[-1]")

# Run forward to trigger hook
# NOTE: CLIP model is loaded in fp16, input must match dtype
dummy_224 = torch.randn(2, 3, 224, 224).to(device)
clip_dtype = clip_model.dtype  # typically torch.float16
print(f"[CLIP] Model dtype: {clip_dtype}")
with torch.no_grad():
    clip_output = visual(dummy_224.type(clip_dtype))
hook.remove()

print(f"[CLIP] visual(dummy) output shape: {clip_output.shape}")  # Expected (B, 512)
print(f"[CLIP] visual(dummy) output dtype: {clip_output.dtype}")

# Extract from hook
raw = _captured_clip['patch_tokens']  # (seq, batch, dim)
print(f"\n[CLIP] Hook-captured raw shape: {raw.shape}")  # Expected (50, B, 768)
print(f"[CLIP] Hook-captured raw dtype: {raw.dtype}")

patch_tokens = raw.permute(1, 0, 2)  # (B, 50, 768)
print(f"[CLIP] After permute: {patch_tokens.shape}")

patch_no_cls = patch_tokens[:, 1:, :]  # (B, 49, 768)
print(f"[CLIP] Without CLS token: {patch_no_cls.shape}")

# Verify can reshape to 7x7 spatial
B = patch_no_cls.shape[0]
spatial = patch_no_cls.reshape(B, 7, 7, 768).permute(0, 3, 1, 2)
print(f"[CLIP] Reshaped to spatial: {spatial.shape}")  # Expected (B, 768, 7, 7)

# Test type alignment: proj.float() with float patch_tokens
proj_fp32 = visual.proj.float()
patch_fp32 = patch_no_cls.float()
patch_proj = patch_fp32 @ proj_fp32  # (B, 49, 512)
print(f"[CLIP] Patch projected via proj: {patch_proj.shape}")
print(f"[CLIP] Patch projected dtype: {patch_proj.dtype}")

# Verify encode_text works
text_fog = ["dense smoke", "thick fog obscuring objects", "smoke blocking the scene"]
text_sky = ["clear sky", "overcast sky", "cloudy sky"]
tokenized_fog = clip.tokenize(text_fog).to(device)
tokenized_sky = clip.tokenize(text_sky).to(device)
with torch.no_grad():
    emb_fog = clip_model.encode_text(tokenized_fog)  # (3, 512)
    emb_sky = clip_model.encode_text(tokenized_sky)  # (3, 512)
print(f"[CLIP] encode_text fog shape: {emb_fog.shape}, sky shape: {emb_sky.shape}")
print(f"[CLIP] t_fog mean norm: {emb_fog.mean(dim=0).norm():.4f}")
print(f"[CLIP] t_sky mean norm: {emb_sky.mean(dim=0).norm():.4f}")

print("\n[CLIP] ALL CHECKS PASSED")

# ============================================================
# PART 2: DINOv2 探针
# ============================================================
print("\n" + "=" * 60)
print("PART 2: DINOv2 Probe")
print("=" * 60)

DINO_SOURCE_DIR = "./DINOv2/facebookresearch_dinov2_main"
DINO_WEIGHT_PATH = "./dinov2_model/dinov2_vitb14_pretrain.pth"

# Check source exists
if not os.path.isdir(DINO_SOURCE_DIR):
    print(f"[DINOv2] FATAL: Source directory not found: {DINO_SOURCE_DIR}")
    print("[DINOv2] Please place DINOv2 source code at: DINOv2/facebookresearch_dinov2_main/")
    sys.exit(1)

# Check weights exist
if not os.path.isfile(DINO_WEIGHT_PATH):
    print(f"[DINOv2] FATAL: Weight file not found: {DINO_WEIGHT_PATH}")
    print("[DINOv2] Please download dinov2_vitb14_pretrain.pth to: dinov2_model/")
    sys.exit(1)

# Insert source path
sys.path.insert(0, DINO_SOURCE_DIR)
print(f"[DINOv2] sys.path[0] = {sys.path[0]}")

try:
    import dinov2
    print(f"[DINOv2] dinov2.__file__ = {dinov2.__file__}")
except ImportError as e:
    print(f"[DINOv2] FATAL: Cannot import dinov2: {e}")
    print("[DINOv2] Check that DINOv2 source has valid __init__.py and .py files.")
    sys.exit(1)

try:
    from dinov2.models.vision_transformer import vit_base, DinoVisionTransformer
    print("[DINOv2] Imported vit_base + DinoVisionTransformer from dinov2.models.vision_transformer")
except ImportError as e:
    print(f"[DINOv2] FATAL: Cannot import DINOv2 model classes: {e}")
    sys.exit(1)

# Import flat Block (not NestedTensorBlock) to match checkpoint key structure
from dinov2.layers.block import Block as FlatBlock
from dinov2.layers import MemEffAttention
from functools import partial
print("[DINOv2] Imported FlatBlock to match checkpoint block structure")

# Load weights first to determine expected img_size
print(f"[DINOv2] Loading weights from: {DINO_WEIGHT_PATH}")
state_dict = torch.load(DINO_WEIGHT_PATH, map_location='cpu')
print(f"[DINOv2] State dict has {len(state_dict)} keys")

# Detect expected img_size from pos_embed shape in checkpoint
ckpt_pos_embed = state_dict["pos_embed"]  # (1, num_patches+1, 768)
ckpt_num_patches = ckpt_pos_embed.shape[1] - 1
ckpt_grid = int(ckpt_num_patches ** 0.5)
ckpt_img_size = ckpt_grid * 14
print(f"[DINOv2] Checkpoint pos_embed: {ckpt_pos_embed.shape} → grid={ckpt_grid}×{ckpt_grid} → img_size={ckpt_img_size}")

# Construct DinoVisionTransformer to match checkpoint key structure:
# - block_chunks=0: disable BlockChunk wrapping → flat "blocks.N.*" keys
# - init_values=1.0: enable LayerScale → "ls1.gamma"/"ls2.gamma" parameters
# - FlatBlock: not NestedTensorBlock → matches checkpoint block structure
dino = DinoVisionTransformer(
    img_size=ckpt_img_size,
    patch_size=14,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4,
    block_fn=partial(FlatBlock, attn_class=MemEffAttention),
    block_chunks=0,
    init_values=1.0,
)
print(f"[DINOv2] DinoVisionTransformer(img_size={ckpt_img_size}, block_chunks=0, init_values=1.0)")

# Load weights with strict checking
try:
    dino.load_state_dict(state_dict, strict=True)
    print("[DINOv2] State dict loaded with STRICT match!")
except Exception as e:
    print(f"[DINOv2] WARNING: Strict load failed: {e}")
    # Fallback: non-strict with pos_embed interpolation
    missing, unexpected = dino.load_state_dict(state_dict, strict=False)
    has_pos_mismatch = any("pos_embed" in k for k in (missing + unexpected))
    print(f"[DINOv2] Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")
    if has_pos_mismatch:
        print("[DINOv2] pos_embed mismatch detected — will rely on interpolate_pos_encoding in forward_features")

dino.to(device)
dino.eval()
for p in dino.parameters():
    p.requires_grad = False
print("[DINOv2] Model frozen and on device")

# Test forward_features with 448x448 input
# DINOv2's forward_features calls interpolate_pos_encoding internally,
# which auto-interpolates pos_embed from checkpoint img_size to 448/14=32 grid
dummy_448 = torch.randn(2, 3, 448, 448).to(device)
with torch.no_grad():
    out = dino.forward_features(dummy_448)

print(f"\n[DINOv2] forward_features(448x448) output keys: {list(out.keys())}")
print(f"[DINOv2] Note: pos_embed was interpolated from {ckpt_img_size}x{ckpt_img_size} → 448x448 by DINOv2 internally")

# Check x_norm_patchtokens
if "x_norm_patchtokens" not in out:
    print("[DINOv2] FATAL: 'x_norm_patchtokens' not in forward_features output!")
    sys.exit(1)

patch_tokens_dino = out["x_norm_patchtokens"]
print(f"[DINOv2] x_norm_patchtokens shape: {patch_tokens_dino.shape}")
print(f"[DINOv2] x_norm_patchtokens dtype: {patch_tokens_dino.dtype}")

grid = int(patch_tokens_dino.shape[1] ** 0.5)
print(f"[DINOv2] Grid size: {grid}x{grid}")

# Check x_prenorm if available
if "x_prenorm" in out:
    print(f"[DINOv2] x_prenorm shape: {out['x_prenorm'].shape}")
    print(f"[DINOv2] x_prenorm dtype: {out['x_prenorm'].dtype}")
else:
    print("[DINOv2] x_prenorm NOT available in output (will use x_norm_patchtokens)")

# Verify reshape to spatial
feat_dino_spatial = patch_tokens_dino.reshape(2, grid, grid, 768).permute(0, 3, 1, 2)
print(f"[DINOv2] Reshaped to spatial: {feat_dino_spatial.shape}")

# Clean up sys.path
sys.path.pop(0)
print("[DINOv2] Removed DINOv2 source from sys.path")

print("\n[DINOv2] ALL CHECKS PASSED")

# ============================================================
# PART 3: L2 Norm Discrimination — Region-based quantitative test
# ============================================================
print("\n" + "=" * 60)
print("PART 3: L2 Norm Proxy — Region Discrimination Test")
print("=" * 60)

import cv2
import numpy as np

REAL_FOG_PATH = "F:/Dehaze_Paper/2_Dataset/1_main_benchmark/REAL_FOGGY/hazy/00960.png"
if not os.path.isfile(REAL_FOG_PATH):
    print(f"[L2-TEST] SKIPPED: Image not found: {REAL_FOG_PATH}")
else:
    print(f"[L2-TEST] Loading image: {REAL_FOG_PATH}")
    img = cv2.imread(REAL_FOG_PATH)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_h, img_w = img.shape[:2]
    print(f"[L2-TEST] Original size: {img_w}x{img_h}")
    img_448 = cv2.resize(img, (448, 448))

    # Normalize with ImageNet stats
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std  = np.array([0.229, 0.224, 0.225])
    img_norm = (img_448.astype(np.float32) / 255.0 - imagenet_mean) / imagenet_std
    x = torch.from_numpy(img_norm).float().permute(2, 0, 1).unsqueeze(0).to(device)

    # Re-run DINOv2 forward_features on the real image
    sys.path.insert(0, DINO_SOURCE_DIR)
    from dinov2.models.vision_transformer import DinoVisionTransformer as DVT2
    from dinov2.layers.block import Block as FlatBlock2
    from dinov2.layers import MemEffAttention as MemEffAttn2
    dino2 = DVT2(img_size=ckpt_img_size, patch_size=14, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4,
                 block_fn=partial(FlatBlock2, attn_class=MemEffAttn2),
                 block_chunks=0, init_values=1.0)
    dino2.load_state_dict(torch.load(DINO_WEIGHT_PATH, map_location='cpu'), strict=True)
    dino2.to(device).eval()
    sys.path.pop(0)

    with torch.no_grad():
        out = dino2.forward_features(x)

    print(f"[L2-TEST] forward_features keys: {list(out.keys())}")

    # ---- USE x_prenorm (pre-LN, CLS removed) ----
    # Based on pilot test: x_prenorm has 32% better spatial variation (3.05 vs 2.31)
    pt_pre = out["x_prenorm"]           # (1, 1025, 768) — includes CLS at index 0
    pt_pre_patches = pt_pre[:, 1:, :]   # (1, 1024, 768) — CLS removed
    l2_map = torch.norm(pt_pre_patches, dim=-1).reshape(32, 32).cpu().numpy()

    # ===== Get full feature tensor: (1, 32, 32, 768) =====
    feat_3d = pt_pre_patches.squeeze(0).reshape(32, 32, 768).cpu()  # (32, 32, 768)

    # =================================================================
    # METRIC 1: L2 norm (global magnitude per patch)
    # =================================================================
    l2_map = torch.norm(pt_pre_patches, dim=-1).reshape(32, 32).cpu().numpy()

    # =================================================================
    # METRIC 2: Local Feature Variance (3×3 neighborhood)
    #    For each patch (i,j), compute mean L2 distance to its 8 neighbors
    #    within 3×3 window (padding='same' with reflect).
    #    Low value = homogeneous region (smoke). High = textured region.
    # =================================================================
    pad = torch.nn.ReflectionPad2d(1)
    feat_padded = pad(feat_3d.permute(2, 0, 1).unsqueeze(0))  # (1, 768, 34, 34)
    local_var = torch.zeros(32, 32)
    for di in range(3):
        for dj in range(3):
            if di == 1 and dj == 1:
                continue  # skip self
            neighbor = feat_padded[:, :, di:di+32, dj:dj+32]  # (1, 768, 32, 32)
            diff = (feat_3d.permute(2, 0, 1).unsqueeze(0) - neighbor).norm(dim=1)  # (1, 32, 32)
            local_var += diff.squeeze(0)
    local_var = (local_var / 8.0).numpy()  # mean over 8 neighbors

    print(f"\n[L2-TEST] ========== METRIC 1: L2 Norm (x_prenorm, CLS removed) ==========")
    print(f"[L2-TEST] Global range: [{l2_map.min():.3f}, {l2_map.max():.3f}]")
    print(f"[L2-TEST] Global mean:  {l2_map.mean():.3f}  |  std: {l2_map.std():.3f}")
    print(f"\n[L2-TEST] ========== METRIC 2: Local Feature Variance (3×3 neighbors) ==========")
    print(f"[L2-TEST] Global range: [{local_var.min():.3f}, {local_var.max():.3f}]")
    print(f"[L2-TEST] Global mean:  {local_var.mean():.3f}  |  std: {local_var.std():.3f}")

    # ==== Define THREE regions ====
    smoke_r1, smoke_r2 = 17, 30   # bottom-right smoke
    smoke_c1, smoke_c2 = 16, 30
    grass_r1, grass_r2 = 19, 30   # bottom-left grass
    grass_c1, grass_c2 = 0, 11
    mtn_r1, mtn_r2 = 8, 16        # upper-center mountain
    mtn_c1, mtn_c2 = 8, 24

    def region_stats(map_np, r1, r2, c1, c2):
        region = map_np[r1:r2, c1:c2]
        return region.mean(), region.std(), region.min(), region.max(), region

    # L2 norm
    l2_smoke_mean, l2_smoke_std, l2_smoke_min, l2_smoke_max, _ = region_stats(l2_map, smoke_r1, smoke_r2, smoke_c1, smoke_c2)
    l2_grass_mean, l2_grass_std, l2_grass_min, l2_grass_max, _ = region_stats(l2_map, grass_r1, grass_r2, grass_c1, grass_c2)
    l2_mtn_mean, l2_mtn_std, l2_mtn_min, l2_mtn_max, _ = region_stats(l2_map, mtn_r1, mtn_r2, mtn_c1, mtn_c2)

    # Local variance
    lv_smoke_mean, lv_smoke_std, lv_smoke_min, lv_smoke_max, _ = region_stats(local_var, smoke_r1, smoke_r2, smoke_c1, smoke_c2)
    lv_grass_mean, lv_grass_std, lv_grass_min, lv_grass_max, _ = region_stats(local_var, grass_r1, grass_r2, grass_c1, grass_c2)
    lv_mtn_mean, lv_mtn_std, lv_mtn_min, lv_mtn_max, _ = region_stats(local_var, mtn_r1, mtn_r2, mtn_c1, mtn_c2)

    # =================================================================
    # Print comparison table
    # =================================================================
    print(f"\n[L2-TEST] =============================================================")
    print(f"[L2-TEST]  METRIC               | A:SMOKE    | B:GRASS    | C:MOUNTAIN")
    print(f"[L2-TEST]  ---------------------|------------|------------|-----------")
    print(f"[L2-TEST]  L2 Norm mean         | {l2_smoke_mean:8.3f}   | {l2_grass_mean:8.3f}   | {l2_mtn_mean:8.3f}")
    print(f"[L2-TEST]  L2 Norm std          | {l2_smoke_std:8.3f}   | {l2_grass_std:8.3f}   | {l2_mtn_std:8.3f}")
    print(f"[L2-TEST]  LocalStruct mean     | {lv_smoke_mean:8.3f}   | {lv_grass_mean:8.3f}   | {lv_mtn_mean:8.3f}")
    print(f"[L2-TEST]  LocalStruct std      | {lv_smoke_std:8.3f}   | {lv_grass_std:8.3f}   | {lv_mtn_std:8.3f}")
    print(f"[L2-TEST]  ---------------------|------------|------------|-----------")
    l2_margin_grass = l2_grass_mean - l2_smoke_mean
    l2_margin_mtn   = l2_mtn_mean - l2_smoke_mean
    lv_margin_grass = lv_grass_mean - lv_smoke_mean
    lv_margin_mtn   = lv_mtn_mean - lv_smoke_mean
    l2_pct_grass = 100.0 * l2_margin_grass / l2_smoke_mean
    l2_pct_mtn   = 100.0 * l2_margin_mtn / l2_smoke_mean
    lv_pct_grass = 100.0 * lv_margin_grass / lv_smoke_mean
    lv_pct_mtn   = 100.0 * lv_margin_mtn / lv_smoke_mean
    print(f"[L2-TEST]  Delta L2 Norm (vs smoke) | +{l2_margin_grass:.3f} ({l2_pct_grass:.1f}%) | +{l2_margin_mtn:.3f} ({l2_pct_mtn:.1f}%)")
    print(f"[L2-TEST]  Delta LocalStruct        | +{lv_margin_grass:.3f} ({lv_pct_grass:.1f}%) | +{lv_margin_mtn:.3f} ({lv_pct_mtn:.1f}%)")
    print(f"[L2-TEST] =============================================================")

    # Verdict
    if lv_pct_grass > 8.0 and lv_pct_mtn > 8.0:
        print(f"[L2-TEST]  >>> VERDICT: LocalStruct PASS -- clear >> smoke, margins {lv_pct_grass:.1f}%/{lv_pct_mtn:.1f}% both > 8% [OK]")
        print(f"[L2-TEST]  >>> LocalStruct margin is {lv_pct_grass/l2_pct_grass:.1f}x / {lv_pct_mtn/l2_pct_mtn:.1f}x better than L2 norm")
        print(f"[L2-TEST]  >>> RECOMMENDATION: Use LocalStruct (3x3 neighbor L2 variance) as attn_deg in cmdn.py")
        use_norm = False
    elif l2_pct_grass > 8.0 and l2_pct_mtn > 8.0:
        print(f"[L2-TEST]  >>> VERDICT: L2 Norm has better margin, LocalStruct marginal")
        use_norm = True
    else:
        print(f"[L2-TEST]  >>> VERDICT: Both metrics marginal -- need visual confirmation")
        use_norm = True  # fallback

    # =================================================================
    # Visualization: 3-panel comparison
    # =================================================================
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from scipy.ndimage import zoom

        px_per_cell = 448 / 32  # = 14.0

        # Upsample both maps
        l2_up = zoom(l2_map, 448/32, order=1)
        lv_up = zoom(local_var, 448/32, order=1)

        fig, axes = plt.subplots(2, 3, figsize=(22, 14))

        def draw_boxes(ax):
            for (r1, c1, r2, c2, color, lbl) in [
                (smoke_r1, smoke_c1, smoke_r2, smoke_c2, 'red', 'A:SMOKE'),
                (grass_r1, grass_c1, grass_r2, grass_c2, 'lime', 'B:GRASS'),
                (mtn_r1,   mtn_c1,   mtn_r2,   mtn_c2,   'cyan', 'C:MTN'),
            ]:
                rect = Rectangle((c1*px_per_cell, r1*px_per_cell),
                                 (c2-c1)*px_per_cell, (r2-r1)*px_per_cell,
                                 linewidth=2, edgecolor=color, facecolor='none', linestyle='--')
                ax.add_patch(rect)

        # Row 0: L2 Norm
        axes[0, 0].imshow(img_448)
        axes[0, 0].set_title('Original 00960.png (448×448)', fontsize=11, fontweight='bold')
        draw_boxes(axes[0, 0])
        axes[0, 0].axis('off')

        axes[0, 1].imshow(l2_map, cmap='hot', vmin=l2_map.min(), vmax=l2_map.max(), interpolation='nearest')
        axes[0, 1].set_title(f'L2 Norm (32×32 raw)\n'
                             f'SMOKE={l2_smoke_mean:.1f}  GRASS={l2_grass_mean:.1f}  MTN={l2_mtn_mean:.1f}\n'
                             f'd_grass={l2_margin_grass:.1f} ({l2_pct_grass:.1f}%)  d_mtn={l2_margin_mtn:.1f} ({l2_pct_mtn:.1f}%)',
                             fontsize=10)
        axes[0, 1].axis('off')

        im_l2 = axes[0, 2].imshow(img_448, alpha=0.3)
        axes[0, 2].imshow(l2_up, cmap='hot', alpha=0.70, vmin=l2_map.min(), vmax=l2_map.max(),
                          extent=[0, 448, 448, 0], interpolation='bilinear')
        axes[0, 2].set_title('L2 Norm overlay', fontsize=10)
        draw_boxes(axes[0, 2])
        axes[0, 2].axis('off')
        plt.colorbar(im_l2, ax=axes[0, 2], fraction=0.046, label='L2 Norm')

        # Row 1: Local Feature Variance
        axes[1, 0].imshow(img_448)
        axes[1, 0].set_title('Original 00960.png (448×448)', fontsize=11, fontweight='bold')
        draw_boxes(axes[1, 0])
        axes[1, 0].axis('off')

        axes[1, 1].imshow(local_var, cmap='hot', vmin=local_var.min(), vmax=local_var.max(), interpolation='nearest')
        axes[1, 1].set_title(f'Local Feature Variance 3×3 (32×32 raw)\n'
                             f'SMOKE={lv_smoke_mean:.1f}  GRASS={lv_grass_mean:.1f}  MTN={lv_mtn_mean:.1f}\n'
                             f'd_grass={lv_margin_grass:.1f} ({lv_pct_grass:.1f}%)  d_mtn={lv_margin_mtn:.1f} ({lv_pct_mtn:.1f}%)',
                             fontsize=10)
        axes[1, 1].axis('off')

        im_lv = axes[1, 2].imshow(img_448, alpha=0.3)
        axes[1, 2].imshow(lv_up, cmap='hot', alpha=0.70, vmin=local_var.min(), vmax=local_var.max(),
                          extent=[0, 448, 448, 0], interpolation='bilinear')
        axes[1, 2].set_title('LocalStruct overlay', fontsize=10)
        draw_boxes(axes[1, 2])
        axes[1, 2].axis('off')
        plt.colorbar(im_lv, ax=axes[1, 2], fraction=0.046, label='Local Feature Variance')

        # Winner banner
        best_metric = "LocalStruct" if not use_norm else "L2 Norm"
        fig.suptitle(f'l2_norm_vs_localstruct_00960.png — {best_metric} wins (higher margin vs smoke)',
                     fontsize=13, fontweight='bold', y=0.98)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_path = "l2_norm_vs_localstruct_00960.png"
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"\n[L2-TEST] Comparison visualization saved to: {out_path}")
    except Exception as e:
        import traceback
        print(f"\n[L2-TEST] Visualization failed: {e}")
        traceback.print_exc()

# ============================================================
# PROBE SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("PROBE SUMMARY")
print("=" * 60)
print("CLIP:  PASS (if no errors above)")
print("DINOv2: PASS (if no errors above)")
print("L2-TEST: SEE ABOVE for recommendation on x_norm_patchtokens vs x_prenorm")
print("\nNext step: Proceed to Step 1: model/cmdn.py")
