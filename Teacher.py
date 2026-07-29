"""Source-domain training entry point for fog-routed RGB--TIR dehazing."""

import argparse
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.data_loader import SynthMultiModalDataset, collate_synth
from option.Teacher import (build_parser, persisted_config_from_args, prepare_experiment_dirs,
                            save_config, tir_normalization_config_from_args, validate_config)
from training.source import compute_physical_mask_batch_losses
from training.schedule import build_coa_adam, cycle_batches, set_cosine_learning_rate
from training.observability import TrainingLogger
from training.validation import evaluate_paired_validation, save_best_if_improved
from utils.checkpoint import (build_model_from_config, build_source_checkpoint, load_source_checkpoint,
                              preflight_source_resume_checkpoint)
from utils.metrics import psnr, ssim_global
from utils.visualize_fog_routed import build_diagnostic_panel


SOURCE_RUNTIME_KEYS = (
    "train_data_dir", "validation_data_dir", "resume_checkpoint", "device", "epochs", "iters_per_epoch", "start_lr", "end_lr", "no_lr_sche",
    "batch_size", "validation_batch_size", "num_workers", "exp_dir", "saved_model_dir", "saved_data_dir",
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
        resume_preflight = preflight_source_resume_checkpoint(resume)
        checkpoint_config = resume_preflight["config"]
        resumed_config = resolve_source_resume_config(raw_args, checkpoint_config)
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
    validation_dataset = SynthMultiModalDataset(
        args.validation_data_dir, train=False, size="full", density_gt_semantics=args.density_gt_semantics,
        density_map_normalization=args.density_map_normalization, density_fixed_min=args.density_fixed_min,
        density_fixed_max=args.density_fixed_max, density_calibrated_min=args.density_calibrated_min,
        density_calibrated_max=args.density_calibrated_max, tir_normalization_config=tir_normalization_config_from_args(args),
        pair_alignment_policy=args.pair_alignment_policy,
    )
    validation_loader = DataLoader(validation_dataset, batch_size=args.validation_batch_size, shuffle=False,
                                   num_workers=args.num_workers, collate_fn=collate_synth)
    model = build_model_from_config(config).to(device)
    optimizer = build_coa_adam(model.parameters(), learning_rate=args.start_lr)
    start_epoch, global_step, best_psnr = 0, 0, float("-inf")
    if resume is not None:
        restored = load_source_checkpoint(resume, model, optimizer)
        start_epoch, global_step, best_psnr = restored["epoch"], restored["global_step"], restored.get("best_psnr", float("-inf"))
    total_steps = args.epochs * args.iters_per_epoch
    batch_iterator = cycle_batches(loader)
    logger = TrainingLogger(args.exp_dir or args.saved_model_dir, resume=resume is not None)
    logger.write_run_summary({
        "stage": "source", "device": str(device), "seed": args.model_init_seed,
        "dataset_sizes": {"train": len(dataset), "validation": len(validation_dataset)},
        "batch_configuration": {"train": args.batch_size, "validation": args.validation_batch_size,
                                "num_workers": args.num_workers},
        "schedule": {"epochs": args.epochs, "iters_per_epoch": args.iters_per_epoch,
                     "total_steps": total_steps, "start_lr": args.start_lr, "end_lr": args.end_lr},
        "parameter_counts": {"rgb_encoder": sum(parameter.numel() for parameter in model.rgb_encoder.parameters()),
                             "tir_encoder": sum(parameter.numel() for parameter in model.tir_encoder.parameters()),
                             "total": sum(parameter.numel() for parameter in model.parameters())},
        "res2net_pretrained_loaded": True,
        "res2net_pretrained_path": model.rgb_encoder.pretrained_path,
        "output_directories": {"experiment": args.exp_dir, "models": args.saved_model_dir,
                               "diagnostics": args.saved_data_dir},
    })
    print(f"source startup device={device} train={len(dataset)} validation={len(validation_dataset)} "
          f"epochs={args.epochs} iters_per_epoch={args.iters_per_epoch} total_steps={total_steps} "
          f"params={sum(parameter.numel() for parameter in model.parameters()):,} "
          f"res2net={model.rgb_encoder.pretrained_path}")
    for epoch in range(start_epoch, args.epochs):
        dataset.set_sampler_epoch(epoch)
        model.train()
        logical_step = 0
        while logical_step < args.iters_per_epoch:
            step_started = perf_counter()
            hazy, clear, tir, density, completion_mask = next(batch_iterator)
            if hazy.numel() == 0:
                continue
            schedule_step = epoch * args.iters_per_epoch + logical_step + 1
            learning_rate = set_cosine_learning_rate(
                optimizer, step=schedule_step, total_steps=total_steps, start_lr=args.start_lr,
                end_lr=args.end_lr, no_lr_sche=args.no_lr_sche,
            )
            hazy, clear, tir, density, completion_mask = (value.to(device) for value in (hazy, clear, tir, density, completion_mask))
            optimizer.zero_grad(set_to_none=True)
            result = compute_physical_mask_batch_losses(model, (hazy, clear, tir, density, completion_mask), args, global_step)
            loss = result["losses"]["total"]
            _require_finite_loss(loss, epoch=epoch, step=global_step)
            loss.backward()
            _require_finite_gradients(model, epoch=epoch, step=global_step, loss=loss)
            optimizer.step()
            global_step += 1
            logical_step += 1
            train_psnr = psnr(result["output"]["pred_clear"].detach(), clear).item()
            train_ssim = ssim_global(result["output"]["pred_clear"].detach(), clear).item()
            logger.log_event(
                "train_step", epoch=epoch + 1, epoch_step=logical_step, global_step=global_step,
                learning_rate=learning_rate, duration_seconds=perf_counter() - step_started,
                train_psnr=train_psnr, train_ssim=train_ssim,
                **{f"loss_{name}": value for name, value in result["losses"].items()},
                **{f"schedule_{name}": value for name, value in result["state"].items()},
            )
            if global_step % 50 == 0:
                print(f"source epoch={epoch + 1} step={global_step} loss={loss.item():.5f} "
                      f"reconstruction={result['losses']['reconstruction'].item():.5f} route={result['losses']['route'].item():.5f} "
                      f"lr={learning_rate:.9f} psnr={train_psnr:.3f} ssim={train_ssim:.4f}")
            if args.saved_data_dir and global_step % 500 == 0:
                build_diagnostic_panel(hazy, tir, clear, density, completion_mask, result["output"]).save(
                    Path(args.saved_data_dir) / f"source_step_{global_step:08d}.png"
                )
        validation = evaluate_paired_validation(model, validation_loader, device, route_temperature=args.route_tau_end)
        candidate = build_source_checkpoint(model, optimizer, epoch=epoch + 1, global_step=global_step,
                                            config=config, best_psnr=max(best_psnr, validation["psnr"]))
        is_best = validation["psnr"] > best_psnr
        if args.saved_model_dir:
            output_dir = Path(args.saved_model_dir)
            best_psnr = save_best_if_improved(validation["psnr"], best_psnr, candidate, output_dir / "source_best.pt")
            candidate["best_psnr"] = best_psnr
            torch.save(candidate, output_dir / "source_last.pt")
        logger.log_event("validation", epoch=epoch + 1, global_step=global_step, learning_rate=learning_rate,
                         psnr=validation["psnr"], ssim=validation["ssim"], best_psnr=best_psnr, is_best=is_best)
        print(f"source validation epoch={epoch + 1} psnr={validation['psnr']:.4f} "
              f"ssim={validation['ssim']:.4f} best_psnr={best_psnr:.4f} is_best={is_best}")
    return model


if __name__ == "__main__":
    main()
