"""Formal source-domain training entry point for fog-routed RGB--TIR dehazing."""

import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data import StatefulRandomSampler
from data.stateful_sampler import validate_single_process_world
from data.data_loader import SynthMultiModalDataset, collate_synth
from option.Teacher import build_parser, prepare_experiment_dirs, save_config, validate_config
from option._formal_config import tir_normalization_config_from_args
from training.omega_sampler import OmegaSampler
from training.source_step import compute_source_batch_losses
from training.step_control import perform_optimizer_step
from training.omega_state import update_empty_omega_streak
from training.metrics import psnr, ssim_global
from utils.visualize_fog_routed import build_diagnostic_panel
from training.checkpointing import (
    build_source_checkpoint, build_formal_model_from_config, capture_rng_state,
    restore_source_training_state, validate_checkpoint_metadata, validate_model_semantics_config,
)
from training.resume_config import validate_source_resume_semantics


def _set_model_init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(argv=None):
    args = validate_config(build_parser().parse_args(argv))
    validate_single_process_world()
    prepare_experiment_dirs(args)
    save_config(args)
    _set_model_init_seed(args.model_init_seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    resume_checkpoint = torch.load(args.resume_checkpoint, map_location="cpu") if args.resume_checkpoint else None
    if resume_checkpoint is not None:
        validate_checkpoint_metadata(resume_checkpoint, "source", args.density_gt_semantics)
        validate_model_semantics_config(resume_checkpoint["config"], vars(args))
        validate_source_resume_semantics(resume_checkpoint["config"], vars(args))
        if resume_checkpoint["config"].get("train_size") != args.train_size:
            raise ValueError("source checkpoint train_size/preprocessing mismatch")
    dataset = SynthMultiModalDataset(
        args.train_data_dir, train=True, size=args.train_size, density_gt_semantics=args.density_gt_semantics,
        density_map_normalization=args.density_map_normalization,
        density_fixed_min=args.density_fixed_min, density_fixed_max=args.density_fixed_max,
        density_calibrated_min=args.density_calibrated_min, density_calibrated_max=args.density_calibrated_max,
        tir_normalization_config=tir_normalization_config_from_args(args),
        pair_alignment_policy=args.pair_alignment_policy,
        augmentation_seed_base=args.model_init_seed,
    )
    inspection = dataset.inspect_density_sample(0)
    print(f"[density] semantics={args.density_gt_semantics} inspection={inspection}")
    sampler = StatefulRandomSampler(len(dataset), seed=args.model_init_seed)
    dataset.set_sampler_epoch(sampler.epoch)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.num_workers, collate_fn=collate_synth)
    # The same persisted semantic configuration is used for source training,
    # resume and evaluation; no architecture default may silently replace a
    # user-supplied model/preprocessing setting.
    model = build_formal_model_from_config(vars(args)).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    omega_sampler = OmegaSampler(
        regions_per_image=args.omega_regions_per_image,
        min_area=args.omega_min_area,
        max_area=args.omega_max_area,
        seed=args.model_init_seed,
    )
    omega_generator = torch.Generator(device=device)
    omega_generator.manual_seed(args.model_init_seed + 101)
    model.train()
    global_step, empty_omega_streak, start_epoch = 0, 0, 0
    density_batch_logged = False
    if resume_checkpoint is not None:
        restored = restore_source_training_state(
            resume_checkpoint, model, optimizer, sampler, args.density_gt_semantics,
            manifest_fingerprint=dataset.manifest_fingerprint(), omega_generator=omega_generator,
        )
        global_step = restored["global_step"]
        empty_omega_streak = restored["empty_omega_streak"]
        start_epoch = restored["epoch"]
        dataset.set_sampler_epoch(sampler.epoch)
        if sampler.next_sample_position >= len(dataset) and start_epoch < args.epochs:
            # Checkpoints are committed only at successful-step boundaries.
            # At an epoch boundary the next iterator must start from the next
            # saved deterministic permutation, never from an exhausted cursor.
            sampler.advance_epoch()
            dataset.set_sampler_epoch(sampler.epoch)
    for _epoch in range(start_epoch, args.epochs):
        for hazy, clear, tir, density in loader:
            if hazy.numel() == 0:
                continue
            hazy, clear, tir, density = (value.to(device) for value in (hazy, clear, tir, density))
            if not density_batch_logged:
                print(
                    "[density] first_batch converted "
                    f"min={density.min().item():.6f} max={density.max().item():.6f} "
                    f"mean={density.mean().item():.6f}"
                )
                density_batch_logged = True
            source_result = compute_source_batch_losses(
                model, (hazy, clear, tir, density), args, omega_sampler, global_step,
                omega_generator=omega_generator,
            )
            losses, state = source_result["losses"], source_result["state"]
            valid_omega_count = int(source_result["omega"]["omega_support"].shape[0])
            optimizer.zero_grad(set_to_none=True)
            losses["total"].backward()
            succeeded = perform_optimizer_step(optimizer, model.parameters())
            if not succeeded:
                # The committed sampler cursor deliberately remains unchanged.
                # Stop this iterator so a fresh iterator replays the exact
                # uncommitted batch instead of committing a later batch under
                # the earlier cursor position.
                print(f"step={global_step} skipped: non-finite gradients; batch will be retried")
                break
            if succeeded:
                sampler.commit(hazy.shape[0])
                empty_omega_streak = update_empty_omega_streak(
                    empty_omega_streak,
                    effective_lambda_route=args.lambda_router * state["lambda_route"],
                    valid_omega_count=valid_omega_count,
                    step_succeeded=True,
                )
                if empty_omega_streak >= args.max_consecutive_empty_omega_steps:
                    statuses = source_result["omega"]["status"]
                    raise RuntimeError(
                        "route loss remained enabled without valid Omega regions; "
                        f"streak={empty_omega_streak}, statuses={statuses}, "
                        f"density_min={density.min().item():.5f}, density_max={density.max().item():.5f}, "
                        f"density_mean={density.mean().item():.5f}, density_std={density.std().item():.5f}, "
                        f"omega_area=[{args.omega_min_area},{args.omega_max_area}], "
                        f"effective_lambda_route={args.lambda_router * state['lambda_route']:.6f}"
                    )
                global_step += 1
                if global_step % 50 == 0:
                    print(
                        f"step={global_step} total={losses['total'].item():.5f} "
                        f"density={losses['density'].item():.5f} route={losses['route'].item():.5f} "
                        f"route_mean={source_result['output']['route_soft'].mean().item():.4f} "
                        f"psnr={psnr(source_result['output']['pred_clear'].detach(), clear).item():.3f} "
                        f"ssim={ssim_global(source_result['output']['pred_clear'].detach(), clear).item():.4f}"
                    )
                if args.saved_data_dir and global_step % 500 == 0:
                    panel = build_diagnostic_panel(
                        hazy, tir, clear, density, source_result["output"],
                        q=source_result["q"],
                    )
                    panel.save(Path(args.saved_data_dir) / f"source_step_{global_step:08d}.png")
        if sampler.next_sample_position >= len(dataset) and _epoch + 1 < args.epochs:
            sampler.advance_epoch()
            dataset.set_sampler_epoch(sampler.epoch)
        if args.saved_model_dir:
            checkpoint = build_source_checkpoint(
                model.state_dict(), optimizer.state_dict(), None, global_step, _epoch + 1,
                vars(args), args.density_gt_semantics,
                capture_rng_state(omega_generator=omega_generator),
                sampler_states={"source": sampler.state_dict()},
                empty_omega_streaks={"source": empty_omega_streak},
                manifest_fingerprints={"source": dataset.manifest_fingerprint()},
            )
            torch.save(checkpoint, Path(args.saved_model_dir) / "source_last.pt")


if __name__ == "__main__":
    main()
