import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader

from data.data_loader import SynthMultiModalDataset, collate_synth


root = r"F:/Dehaze_Paper/2_Dataset/1_main_benchmark/FLIR/train"
ds = SynthMultiModalDataset(root=root, train=True, size=256)
print("dataset size:", len(ds))

loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_synth)
hazy, clear, ir, density, mask = next(iter(loader))

print("hazy   ", tuple(hazy.shape), hazy.dtype)
print("clear  ", tuple(clear.shape), clear.dtype)
print("ir     ", tuple(ir.shape), ir.dtype)
print("density", tuple(density.shape), density.dtype, "range", float(density.min()), float(density.max()))
print("mask   ", tuple(mask.shape), mask.dtype, "unique", torch.unique(mask)[:5].tolist())

assert len(ds) > 0
assert hazy.shape[1] == 3 and clear.shape[1] == 3 and ir.shape[1] == 3
assert density.shape[1] == 1 and mask.shape[1] == 1
assert float(density.min()) >= 0.0 and float(density.max()) <= 1.0
assert set(torch.unique(mask).tolist()).issubset({0.0, 1.0})

print("sample0 density mean:", float(density[0].mean()),
      "| mask positive ratio:", float(mask[0].mean()))
print("[loader] check PASSED")
