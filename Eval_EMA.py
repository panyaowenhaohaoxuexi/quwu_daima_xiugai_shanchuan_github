"""Strict EMA-checkpoint evaluation entry point."""

import argparse
import torch

from Eval import evaluate_directory
from training.checkpointing import build_formal_model_from_config, validate_checkpoint_metadata


def build_parser():
    parser = argparse.ArgumentParser("fog-routed-ema-eval")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--ema_eval_model", choices=("teacher", "student"), default="teacher")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hazy_dir", required=True)
    parser.add_argument("--tir_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--format", default="png")
    parser.add_argument("--save_aux", action="store_true")
    return parser


def load_model(checkpoint_path, ema_eval_model="teacher"):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    validate_checkpoint_metadata(checkpoint, "ema", checkpoint.get("density_gt_semantics"))
    config = checkpoint["config"]
    if ema_eval_model not in ("teacher", "student"):
        raise ValueError("ema_eval_model must be 'teacher' or 'student'")
    if ema_eval_model not in checkpoint:
        raise ValueError(f"EMA checkpoint is missing requested {ema_eval_model!r} weights")
    model = build_formal_model_from_config(config)
    model.load_state_dict(checkpoint[ema_eval_model], strict=True)
    return model.eval(), checkpoint


def main(argv=None):
    args = build_parser().parse_args(argv)
    model, checkpoint = load_model(args.checkpoint, args.ema_eval_model)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model.to(device)
    evaluate_directory(model, checkpoint["config"], args.hazy_dir, args.tir_dir,
                       args.output_dir, device, args.save_aux, args.format)
    return model, checkpoint


if __name__ == "__main__":
    main()
