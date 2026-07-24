"""Smoke-check the formal four-tensor synthetic RGB--TIR loader."""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.data_loader import SynthMultiModalDataset, collate_synth


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", required=True, help="synthetic dataset root containing clear/ir/hazy/Transmission_Map_GT")
    parser.add_argument("--train_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--density_gt_semantics", choices=("transmission", "density"), default="transmission")
    parser.add_argument("--pair_alignment_policy", choices=("strict", "resize_tir_to_rgb"), default="strict")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    dataset = SynthMultiModalDataset(
        root=args.data_root,
        train=True,
        size=args.train_size,
        density_gt_semantics=args.density_gt_semantics,
        pair_alignment_policy=args.pair_alignment_policy,
    )
    if not dataset:
        raise RuntimeError(f"no valid synthetic samples found under {args.data_root}")
    print("density inspection:", dataset.inspect_density_sample(0))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_synth)
    hazy, clear, tir, density = next(iter(loader))
    print("hazy   ", tuple(hazy.shape), hazy.dtype, "range", float(hazy.min()), float(hazy.max()))
    print("clear  ", tuple(clear.shape), clear.dtype, "range", float(clear.min()), float(clear.max()))
    print("tir    ", tuple(tir.shape), tir.dtype, "range", float(tir.min()), float(tir.max()))
    print("density", tuple(density.shape), density.dtype, "range", float(density.min()), float(density.max()))
    assert hazy.shape[1] == clear.shape[1] == tir.shape[1] == 3
    assert density.shape[1] == 1 and float(density.min()) >= 0.0 and float(density.max()) <= 1.0
    print("[loader] formal four-tensor check PASSED")


if __name__ == "__main__":
    main()
