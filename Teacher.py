"""Source-domain training entry point for fog-routed RGB--TIR dehazing."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data.data_loader import SynthMultiModalDataset, collate_synth
from option.Teacher import (build_parser, persisted_config_from_args, prepare_experiment_dirs,
                            save_config, tir_normalization_config_from_args, validate_config)
from training.source import OmegaSampler, compute_source_batch_losses
from utils.checkpoint import (build_model_from_config, build_source_checkpoint, load_source_checkpoint,
                              require_checkpoint_format, require_stage)
from utils.metrics import psnr, ssim_global
from utils.visualize_fog_routed import build_diagnostic_panel


SOURCE_RUNTIME_KEYS = (
    "train_data_dir", "resume_checkpoint", "device", "epochs", "learning_rate",
    "batch_size", "num_workers", "exp_dir", "saved_model_dir", "saved_data_dir",
)


def resolve_source_resume_config(raw_args, checkpoint_config):
    """Keep Source semantics from the checkpoint while accepting current runtime settings."""
    raw_config = vars(raw_args)
    semantic_keys = tuple(key for key in raw_config if not key.startswith("_") and key not in SOURCE_RUNTIME_KEYS)
    missing = [key for key in semantic_keys if key not in checkpoint_config]
    if missing:
        raise ValueError("Source checkpoint lacks required training configuration: " + ", ".join(missing))
    return {
        **{key: checkpoint_config[key] for key in semantic_keys},
        **{key: raw_config[key] for key in SOURCE_RUNTIME_KEYS},
    }


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _require_finite_loss(loss, *, epoch, step):
    if not torch.isfinite(loss).all():
        raise RuntimeError(f"source non-finite loss: epoch={epoch} step={step} loss={float(loss.detach())}")


def _require_finite_gradients(model, *, epoch, step, loss):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise RuntimeError(
                f"source non-finite gradient: epoch={epoch} step={step} loss={float(loss.detach())} parameter={name}"
            )


def main(argv=None):
    raw_args = build_parser().parse_args(argv)
    resume = torch.load(raw_args.resume_checkpoint, map_location="cpu") if raw_args.resume_checkpoint else None
    if resume is not None:
        require_checkpoint_format(resume)
        require_stage(resume, "source")
        resumed_config = resolve_source_resume_config(raw_args, resume["config"])
        resumed_config["_resume_checkpoint_semantics"] = True
        args = validate_config(argparse.Namespace(**resumed_config))
    else:
        args = validate_config(raw_args)
    prepare_experiment_dirs(args)
    save_config(args)
    _set_seed(args.model_init_seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    config = persisted_config_from_args(args)
    dataset = SynthMultiModalDataset(
        args.train_data_dir, train=True, size=args.train_size, density_gt_semantics=args.density_gt_semantics,
        density_map_normalization=args.density_map_normalization, density_fixed_min=args.density_fixed_min,
        density_fixed_max=args.density_fixed_max, density_calibrated_min=args.density_calibrated_min,
        density_calibrated_max=args.density_calibrated_max, tir_normalization_config=tir_normalization_config_from_args(args),
        pair_alignment_policy=args.pair_alignment_policy, augmentation_seed_base=args.model_init_seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        collate_fn=collate_synth)
    model = build_model_from_config(config).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    start_epoch, global_step = 0, 0
    if resume is not None:
        restored = load_source_checkpoint(resume, model, optimizer)
        start_epoch, global_step = restored["epoch"], restored["global_step"]
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.learning_rate
    omega_sampler = OmegaSampler(args.omega_regions_per_image, args.omega_min_area,
                                 args.omega_max_area, seed=args.model_init_seed)
    omega_generator = torch.Generator(device=device).manual_seed(args.model_init_seed + 101)
    empty_omega_streak = 0
    for epoch in range(start_epoch, args.epochs):
        dataset.set_sampler_epoch(epoch)
        model.train()
        for hazy, clear, tir, density in loader:
            if hazy.numel() == 0:
                continue
            hazy, clear, tir, density = (value.to(device) for value in (hazy, clear, tir, density))
            optimizer.zero_grad(set_to_none=True)
            result = compute_source_batch_losses(model, (hazy, clear, tir, density), args, omega_sampler, global_step,
                                                 omega_generator=omega_generator)
            loss = result["losses"]["total"]
            _require_finite_loss(loss, epoch=epoch, step=global_step)
            loss.backward()
            _require_finite_gradients(model, epoch=epoch, step=global_step, loss=loss)
            optimizer.step()
            valid = result["route_supervision"]["valid_q_region_count"]
            enabled = args.lambda_router * result["state"]["lambda_route"] > 0
            empty_omega_streak = empty_omega_streak + 1 if enabled and not valid else 0
            if empty_omega_streak >= args.max_consecutive_empty_omega_steps:
                raise RuntimeError(f"source empty Omega limit: epoch={epoch} step={global_step}")
            global_step += 1
            if global_step % 50 == 0:
                print(f"source epoch={epoch + 1} step={global_step} loss={loss.item():.5f} "
                      f"psnr={psnr(result['output']['pred_clear'].detach(), clear).item():.3f} "
                      f"ssim={ssim_global(result['output']['pred_clear'].detach(), clear).item():.4f}")
            if args.saved_data_dir and global_step % 500 == 0:
                build_diagnostic_panel(hazy, tir, clear, density, result["output"], q=result["q"]).save(
                    Path(args.saved_data_dir) / f"source_step_{global_step:08d}.png"
                )
        if args.saved_model_dir:
            torch.save(build_source_checkpoint(model, optimizer, epoch=epoch + 1, global_step=global_step,
                                               config=config), Path(args.saved_model_dir) / "source_last.pt")
    return model


if __name__ == "__main__":
    main()
