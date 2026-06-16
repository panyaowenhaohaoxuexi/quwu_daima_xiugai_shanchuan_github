import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, Normalize, Resize, Compose

from model.cmdn import CMDN


# Real dense-fog visible image and paired infrared image.
VIS_PATH = r"F:\Dehaze_Paper\2_Dataset\1_main_benchmark\FLIR\train\hazy\dense\FLIR_00002.jpg"
IR_PATH = r"F:\Dehaze_Paper\2_Dataset\1_main_benchmark\FLIR\train\ir\FLIR_00002.jpg"
OUT_DIR = r"./md_check_out"


os.makedirs(OUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

clip_norm = Compose([
    Resize((256, 256)),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073),
              (0.26862954, 0.26130258, 0.27577711)),
])


def load(p):
    return clip_norm(Image.open(p).convert("RGB")).unsqueeze(0).to(device)


vis = load(VIS_PATH)
ir = load(IR_PATH)

model = CMDN().to(device).eval()
with torch.no_grad():
    d = model(vis, ir, return_debug=True)


def save_gray(t, name):
    a = t[0, 0].detach().cpu().float().numpy()
    a = (a - a.min()) / (a.max() - a.min() + 1e-6)
    Image.fromarray((a * 255).astype(np.uint8)).save(os.path.join(OUT_DIR, name))
    return float(t.min()), float(t.mean()), float(t.max())


# Save the visible thumbnail as a visual reference.
vis_01 = (
    vis * torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    + torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
).clamp(0, 1)
Image.fromarray((vis_01[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)).save(
    os.path.join(OUT_DIR, "00_vis.png")
)

print("M_d   (min/mean/max):", save_gray(d["M_d"], "01_M_d.png"))
print("C     (min/mean/max):", save_gray(d["C"], "02_C.png"))
print("M_prob(min/mean/max):", save_gray(d["M_prob"], "03_M_prob.png"))
print("M     (min/mean/max):", save_gray(d["M"], "04_M.png"))
print(f"\nSaved to {OUT_DIR}. 人工检查 01_M_d.png：浓雾区应亮、天空区应暗。")
