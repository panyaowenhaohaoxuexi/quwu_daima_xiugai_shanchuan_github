"""Single-pair main-model inference with an offline HDE transmission map.

This script intentionally imports no code and no checkpoint from the HDE
verification project.  It only consumes that project's saved single-channel
transmission image, converts it to the main model's density convention, and
uses the existing source checkpoint for routing and dehazing.
"""

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from data.data_loader import load_scalar_map_as_float_tensor, load_tir_as_float_tensor
from utils.checkpoint import (
    build_model_from_config,
    load_strict_v2_state_dict,
    preflight_eval_checkpoint,
    tir_normalization_config,
)


def build_parser():
    parser = argparse.ArgumentParser("offline-HDE fog-routed single-pair test")
    parser.add_argument("--checkpoint", required=True, help="path to the main source/EMA checkpoint")
    parser.add_argument("--hazy", required=True, help="path to the hazy RGB image")
    parser.add_argument("--tir", required=True, help="path to the TIR image")
    parser.add_argument(
        "--external_transmission_map",
        required=True,
        help="single-channel transmission image produced offline by module_verify_HDE",
    )
    parser.add_argument("--output_dir", required=True, help="directory for output images")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--format", default="png")
    return parser


def load_external_transmission_map(path, expected_size):
    """Load offline raw transmission T and form main-model fog density D=1-T."""
    transmission = load_scalar_map_as_float_tensor(path).unsqueeze(0)
    if tuple(transmission.shape[-2:]) != tuple(expected_size):
        raise ValueError(
            "external transmission size mismatch: "
            f"map={tuple(transmission.shape[-2:])}, rgb={tuple(expected_size)}"
        )
    return transmission, 1.0 - transmission


def inject_external_density(context, density_map, router, route_temperature):
    """Return a shallow context copy with external density and its recomputed route."""
    if tuple(density_map.shape) != tuple(context["density_map"].shape):
        raise ValueError(
            f"external density shape mismatch: map={tuple(density_map.shape)}, "
            f"expected={tuple(context['density_map'].shape)}"
        )
    injected = dict(context)
    injected.update({"density_map": density_map, **router(density_map, route_temperature)})
    return injected


def _load_hazy_rgb(path):
    with Image.open(path) as image:
        return TF.pil_to_tensor(image.convert("RGB")).float().div_(255).unsqueeze(0)


def _save_image(tensor, path):
    TF.to_pil_image(tensor[0].cpu()).save(path)
    print(f"Saved: {path}")


def main(argv=None):
    args = build_parser().parse_args(argv)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    checkpoint_preflight = preflight_eval_checkpoint(checkpoint, ema_model="teacher")
    config = checkpoint_preflight["config"]
    print(f"Checkpoint stage: {checkpoint_preflight['stage']}")
    print(f"State key: {checkpoint_preflight['state_key']}")

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = build_model_from_config(config).to(device)
    load_strict_v2_state_dict(model, checkpoint_preflight["state_dict"], label=checkpoint_preflight["state_label"])
    model.eval()
    print(f"Main model loaded on {device}")

    hazy = _load_hazy_rgb(args.hazy)
    tir = load_tir_as_float_tensor(args.tir, tir_normalization_config(config)).unsqueeze(0)
    policy = config.get("pair_alignment_policy", "strict")
    if tir.shape[-2:] != hazy.shape[-2:]:
        if policy == "strict":
            raise ValueError(f"Size mismatch: hazy={tuple(hazy.shape[-2:])}, tir={tuple(tir.shape[-2:])}")
        tir = F.interpolate(tir, size=hazy.shape[-2:], mode="bilinear", align_corners=False)

    transmission, density = load_external_transmission_map(args.external_transmission_map, hazy.shape[-2:])
    route_temperature = config.get("route_tau_end", 0.2)
    with torch.inference_mode():
        context = model.encode_context(hazy.to(device), tir.to(device), route_temperature=route_temperature)
        context = inject_external_density(context, density.to(device), model.router, route_temperature)
        output = model.decode_with_route(context, route_mode="hard")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "." + str(args.format).lstrip(".")
    stem = Path(args.hazy).stem
    _save_image(transmission, output_dir / f"{stem}_external_transmission{suffix}")
    _save_image(output["density_map"], output_dir / f"{stem}_density_map{suffix}")
    _save_image(output["route_hard"], output_dir / f"{stem}_route_hard{suffix}")
    _save_image(output["route_soft"], output_dir / f"{stem}_route_soft{suffix}")
    _save_image(output["boundary_map"], output_dir / f"{stem}_boundary_map{suffix}")
    _save_image(output["pred_clear"], output_dir / f"{stem}_dehazed{suffix}")
    print("Density source: offline external transmission map (density = 1 - transmission)")
    print(f"Hard completion fraction: {output['route_hard'].mean().item():.6f}")
    print(f"Done! All outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
