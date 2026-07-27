"""Strict source-checkpoint evaluation entry point for the formal model."""

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from data.data_loader import COMMON_IMAGE_EXTS, load_tir_as_float_tensor
from utils.checkpoint import (build_model_from_config, load_strict_v2_state_dict,
                              require_checkpoint_format, require_stage, tir_normalization_config)


def build_parser():
    parser = argparse.ArgumentParser("fog-routed-eval")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--hazy_dir", required=True)
    parser.add_argument("--tir_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--format", default="png")
    parser.add_argument("--save_aux", action="store_true")
    parser.add_argument("--ema_model", choices=("student", "teacher"), default="teacher")
    return parser


def _image_paths(directory):
    return sorted(
        path for path in Path(directory).iterdir()
        if path.is_file() and path.suffix.lower() in COMMON_IMAGE_EXTS
    )


def _unique_stem_paths(directory, *, label):
    index = {}
    for path in _image_paths(directory):
        index.setdefault(path.stem.lower(), []).append(path)
    for stem, candidates in index.items():
        if len(candidates) > 1:
            raise ValueError(
                f"ambiguous {label} stem={stem}: candidates=[{', '.join(str(path) for path in candidates)}]"
            )
    return [candidates[0] for _, candidates in sorted(index.items())]


def _tir_stem_index(tir_dir):
    index = {}
    for path in _image_paths(tir_dir):
        index.setdefault(path.stem.lower(), []).append(path)
    return index


def _resolve_tir_path(index, hazy_path):
    stem = hazy_path.stem.lower()
    candidates = index.get(stem, [])
    if not candidates:
        raise FileNotFoundError(f"missing TIR pair for stem={stem}: hazy_path={hazy_path}")
    if len(candidates) > 1:
        raise ValueError(f"ambiguous TIR pair for stem={stem}: candidates={[str(path) for path in candidates]}")
    return candidates[0]


def evaluate_directory(model, config, hazy_dir, tir_dir, output_dir, device, save_aux=False, image_format="png"):
    """Strict original-resolution inference for Source and EMA checkpoints."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    aux_dir = output_dir / "aux"
    if save_aux:
        aux_dir.mkdir(parents=True, exist_ok=True)
    tir_index = _tir_stem_index(tir_dir)
    with torch.inference_mode():
        for hazy_path in _unique_stem_paths(hazy_dir, label="hazy"):
            tir_path = _resolve_tir_path(tir_index, hazy_path)
            with Image.open(hazy_path) as image:
                hazy = TF.pil_to_tensor(image.convert("RGB")).float().div_(255).unsqueeze(0)
            tir = load_tir_as_float_tensor(tir_path, tir_normalization_config(config)).unsqueeze(0)
            policy = config.get("pair_alignment_policy", "strict")
            if policy not in ("strict", "resize_tir_to_rgb"):
                raise ValueError(f"unsupported checkpoint pair_alignment_policy={policy!r}")
            if tir.shape[-2:] != hazy.shape[-2:]:
                if policy == "strict":
                    raise ValueError(
                        "pair alignment failed "
                        f"policy={policy}, hazy_path={hazy_path}, tir_path={tir_path}, "
                        f"hazy_size={tuple(hazy.shape[-2:])}, tir_size={tuple(tir.shape[-2:])}"
                    )
                tir = F.interpolate(tir, size=hazy.shape[-2:], mode="bilinear", align_corners=False)
            output = model(hazy.to(device), tir.to(device),
                           route_temperature=config["route_tau_end"], route_mode="hard")
            suffix = "." + str(image_format).lstrip(".")
            TF.to_pil_image(output["pred_clear"][0].cpu()).save(output_dir / f"{hazy_path.stem}{suffix}")
            if save_aux:
                stem = hazy_path.stem
                for key in ("density_map", "route_soft", "route_hard", "boundary_map"):
                    TF.to_pil_image(output[key][0].cpu()).save(aux_dir / f"{stem}_{key}.png")


def main(argv=None):
    args = build_parser().parse_args(argv)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    require_checkpoint_format(checkpoint)
    stage = checkpoint.get("training_stage")
    if stage not in ("source", "ema"):
        raise ValueError("checkpoint training_stage must be 'source' or 'ema'")
    config = checkpoint["config"]
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = build_model_from_config(config).to(device)
    load_strict_v2_state_dict(model, checkpoint["model"] if stage == "source" else checkpoint[args.ema_model],
                              label="Source model" if stage == "source" else f"EMA {args.ema_model}")
    model.eval()
    evaluate_directory(model, config, args.hazy_dir, args.tir_dir, args.output_dir, device,
                       args.save_aux, args.format)


if __name__ == "__main__":
    main()
