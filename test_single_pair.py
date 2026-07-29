"""Single-pair inference script for fog-routed RGB--TIR dehazing.

Outputs:
  - pred_clear:        dehazed RGB image (去雾图)
  - density_map:       fog density map (雾气浓度图)
  - route_hard:        hard routing map (路由图)
"""

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from data.data_loader import load_tir_as_float_tensor
from utils.checkpoint import (
    build_model_from_config,
    load_strict_v2_state_dict,
    preflight_eval_checkpoint,
    tir_normalization_config,
)


def build_parser():
    parser = argparse.ArgumentParser("fog-routed-single-pair-test")
    parser.add_argument("--checkpoint", required=True, help="path to source_best.pt")
    parser.add_argument("--hazy", required=True, help="path to hazy RGB image")
    parser.add_argument("--tir", required=True, help="path to TIR/IR image")
    parser.add_argument("--output_dir", required=True, help="directory for output images")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--format", default="png")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    # --- 1. Load checkpoint ---
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_preflight = preflight_eval_checkpoint(checkpoint, ema_model="teacher")
    config = checkpoint_preflight["config"]
    print(f"Checkpoint stage: {checkpoint_preflight['stage']}")
    print(f"State key: {checkpoint_preflight['state_key']}")

    # --- 2. Build model & load weights ---
    device = torch.device(
        args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    )
    model = build_model_from_config(config).to(device)
    load_strict_v2_state_dict(
        model,
        checkpoint_preflight["state_dict"],
        label=checkpoint_preflight["state_label"],
    )
    model.eval()
    print(f"Model loaded on {device}")

    # --- 3. Load images ---
    with Image.open(args.hazy) as image:
        hazy = TF.pil_to_tensor(image.convert("RGB")).float().div_(255).unsqueeze(0)

    tir_config = tir_normalization_config(config)
    tir = load_tir_as_float_tensor(args.tir, tir_config).unsqueeze(0)

    # Align TIR to hazy if needed
    policy = config.get("pair_alignment_policy", "strict")
    if tir.shape[-2:] != hazy.shape[-2:]:
        if policy == "strict":
            raise ValueError(
                f"Size mismatch: hazy={tuple(hazy.shape[-2:])}, tir={tuple(tir.shape[-2:])}"
            )
        tir = F.interpolate(tir, size=hazy.shape[-2:], mode="bilinear", align_corners=False)

    print(f"Hazy shape: {hazy.shape}, TIR shape: {tir.shape}")

    # --- 4. Inference ---
    route_temperature = config.get("route_tau_end", 0.2)
    with torch.inference_mode():
        output = model(
            hazy.to(device),
            tir.to(device),
            route_temperature=route_temperature,
            route_mode="hard",
        )

    # --- 5. Save outputs ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "." + str(args.format).lstrip(".")

    stem = Path(args.hazy).stem

    # 去雾后的图
    dehazed_path = output_dir / f"{stem}_dehazed{suffix}"
    TF.to_pil_image(output["pred_clear"][0].cpu()).save(dehazed_path)
    print(f"Saved: {dehazed_path}")

    # 雾气浓度图
    density_path = output_dir / f"{stem}_density_map{suffix}"
    TF.to_pil_image(output["density_map"][0].cpu()).save(density_path)
    print(f"Saved: {density_path}")

    # 路由图 (hard routing)
    route_path = output_dir / f"{stem}_route_hard{suffix}"
    TF.to_pil_image(output["route_hard"][0].cpu()).save(route_path)
    print(f"Saved: {route_path}")

    # 可选额外输出
    route_soft_path = output_dir / f"{stem}_route_soft{suffix}"
    TF.to_pil_image(output["route_soft"][0].cpu()).save(route_soft_path)
    print(f"Saved: {route_soft_path}")

    boundary_path = output_dir / f"{stem}_boundary_map{suffix}"
    TF.to_pil_image(output["boundary_map"][0].cpu()).save(boundary_path)
    print(f"Saved: {boundary_path}")

    print("\nDone! All outputs saved to:", output_dir)


if __name__ == "__main__":
    main()
