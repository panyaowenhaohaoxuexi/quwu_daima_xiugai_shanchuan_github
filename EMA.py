"""Formal EMA adaptation entry point; no legacy model or CLIP dependency."""

import copy
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data import RealMultiModalDataset, StatefulRandomSampler, SynthMultiModalDataset, collate_real, collate_synth
from data.stateful_sampler import validate_single_process_world
from option.EMA import build_parser, prepare_experiment_dirs, save_config, validate_config
from option._formal_config import tir_normalization_config_from_args
from training.paired_geometry import sample_geometry
from training.ema_core import real_consistency_loss, stability_weights
from training.checkpointing import (
    build_ema_checkpoint, build_formal_model_from_config, capture_rng_state,
    restore_ema_training_state, validate_checkpoint_metadata,
)
from training.ema_step import run_ema_adaptation_step
from training.omega_sampler import OmegaSampler
from training.omega_state import update_empty_omega_streak
from training.source_step import compute_source_batch_losses
from training.resume_config import apply_checkpoint_semantics, apply_ema_resume_config


def set_batchnorm_eval(module):
    for child in module.modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            child.eval()


def initialize_teacher(student):
    teacher = copy.deepcopy(student)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


def run_ema_views(teacher, student, hazy_rgb, tir, generator, route_temperature,
                  sigma_j, sigma_m, sigma_r, minimum_weight):
    """Run independent batch-level A/B/S views and return aligned EMA losses."""
    transform_a, transform_b, transform_s = (sample_geometry(generator) for _ in range(3))
    with torch.no_grad():
        output_a = teacher(transform_a.apply(hazy_rgb), transform_a.apply(tir),
                           route_temperature=route_temperature, route_mode="hard")
        output_b = teacher(transform_b.apply(hazy_rgb), transform_b.apply(tir),
                           route_temperature=route_temperature, route_mode="hard")
    output_s = student(transform_s.apply(hazy_rgb), transform_s.apply(tir),
                       route_temperature=route_temperature, route_mode="hard")
    j_a, j_b = transform_a.inverse(output_a["pred_clear"]), transform_b.inverse(output_b["pred_clear"])
    m_a, m_b = transform_a.inverse(output_a["density_map"]), transform_b.inverse(output_b["density_map"])
    r_a, r_b = transform_a.inverse(output_a["route_soft"]), transform_b.inverse(output_b["route_soft"])
    j_s = transform_s.inverse(output_s["pred_clear"])
    m_s = transform_s.inverse(output_s["density_map"])
    r_s = transform_s.inverse(output_s["route_soft"])
    weights = stability_weights(j_a, j_b, m_a, m_b, r_a, r_b, sigma_j, sigma_m, sigma_r, minimum_weight)
    losses = real_consistency_loss(j_s, 0.5 * (j_a + j_b), m_s, 0.5 * (m_a + m_b),
                                   r_s, 0.5 * (r_a + r_b), *weights)
    return losses, (transform_a, transform_b, transform_s)


def run_ema_epoch(student, teacher, optimizer, real_loader, source_loader, args,
                  source_global_step=0, ema_global_step=0, real_sampler=None, source_sampler=None,
                  anchor_empty_omega_streak=0, geometry_generator=None, omega_generator=None):
    """Run one EMA epoch with independent real/source iterators.

    Every real batch obtains exactly one source anchor.  The source anchor uses
    the fixed end-temperature hard route while retaining continuous soft-route
    losses/q inside the shared source-step implementation.
    """
    device = next(student.parameters()).device
    if geometry_generator is None:
        geometry_generator = torch.Generator(device=device).manual_seed(args.model_init_seed + ema_global_step)
    if omega_generator is None:
        omega_generator = torch.Generator(device=device).manual_seed(args.model_init_seed + 101)
    omega_sampler = OmegaSampler(args.omega_regions_per_image, args.omega_min_area,
                                 args.omega_max_area, seed=args.model_init_seed)
    student.train()
    set_batchnorm_eval(student)
    teacher.eval()
    set_batchnorm_eval(teacher)
    source_iterator = iter(source_loader)
    for real_batch in real_loader:
        try:
            source_batch = next(source_iterator)
        except StopIteration:
            if source_sampler is not None and source_sampler.next_sample_position >= source_sampler.data_source_size:
                source_sampler.advance_epoch()
                source_loader.dataset.set_sampler_epoch(source_sampler.epoch)
            source_iterator = iter(source_loader)
            source_batch = next(source_iterator)
        real_hazy, real_tir, _metadata = real_batch
        real_hazy, real_tir = real_hazy.to(device), real_tir.to(device)
        source_batch = tuple(value.to(device) for value in source_batch)

        def real_loss_fn():
            losses, _ = run_ema_views(
                teacher, student, real_hazy, real_tir, geometry_generator, args.route_tau_end,
                args.ema_sigma_j, args.ema_sigma_m, args.ema_sigma_r, args.ema_stability_min_weight,
            )
            return (args.lambda_ema_j * losses["L_J"] + args.lambda_ema_m * losses["L_M"] +
                    args.lambda_ema_r * losses["L_R"])

        anchor_result = {}

        def anchor_loss_fn():
            result = compute_source_batch_losses(
                student, source_batch, args, omega_sampler, source_global_step,
                force_anchor_mode=True, omega_generator=omega_generator,
            )
            anchor_result["value"] = result
            return result["losses"]["total"]

        result = run_ema_adaptation_step(
            student, teacher, optimizer, real_loss_fn, anchor_loss_fn,
            lambda_anchor=args.lambda_anchor, ema_decay=args.ema_decay,
        )
        if not result["step_succeeded"]:
            # Neither sampler cursor is committed.  Leave this iterator so
            # the next epoch/restart begins from the same uncommitted pair.
            print(f"ema_step={ema_global_step} skipped: non-finite gradients; pair will be retried")
            break
        if result["step_succeeded"]:
            if real_sampler is not None:
                real_sampler.commit(real_hazy.shape[0])
            if source_sampler is not None:
                source_sampler.commit(source_batch[0].shape[0])
            source_global_step += 1
            ema_global_step += 1
            source_result = anchor_result["value"]
            anchor_empty_omega_streak = update_empty_omega_streak(
                anchor_empty_omega_streak,
                effective_lambda_route=args.lambda_router * source_result["state"]["lambda_route"],
                valid_omega_count=int(source_result["omega"]["omega_support"].shape[0]),
                step_succeeded=True,
            )
            if anchor_empty_omega_streak >= args.max_consecutive_empty_omega_steps:
                raise RuntimeError(
                    "EMA source anchor route supervision remained unavailable; "
                    f"streak={anchor_empty_omega_streak}, statuses={source_result['omega']['status']}, "
                    f"effective_lambda_route={args.lambda_router * source_result['state']['lambda_route']:.6f}"
                )
            if ema_global_step % 50 == 0:
                print(
                    f"ema_step={ema_global_step} source_step={source_global_step} "
                    f"L_real={result['L_real'].item():.5f} L_src={result['L_src'].item():.5f} "
                    f"anchor={args.lambda_anchor * result['L_src'].item():.5f}"
                )
    return source_global_step, ema_global_step, anchor_empty_omega_streak


def main(argv=None):
    args = validate_config(build_parser().parse_args(argv))
    validate_single_process_world()
    prepare_experiment_dirs(args)
    random.seed(args.model_init_seed)
    np.random.seed(args.model_init_seed)
    torch.manual_seed(args.model_init_seed)
    resume_checkpoint = torch.load(args.resume_checkpoint, map_location="cpu") if args.resume_checkpoint else None
    if resume_checkpoint is None and not args.source_checkpoint:
        raise ValueError("EMA adaptation requires --source_checkpoint or --resume_checkpoint")
    source_checkpoint = None
    if resume_checkpoint is not None:
        validate_checkpoint_metadata(resume_checkpoint, "ema", resume_checkpoint.get("density_gt_semantics"))
        model_config = resume_checkpoint["config"]
        resolved, config_diff = apply_ema_resume_config(
            vars(args), model_config, allow_training_override=args.allow_ema_training_override
        )
        args = validate_config(argparse.Namespace(**resolved))
        if config_diff:
            print(f"[EMA] explicit training override: {config_diff}")
    else:
        source_checkpoint = torch.load(args.source_checkpoint, map_location="cpu")
        validate_checkpoint_metadata(source_checkpoint, "source", source_checkpoint.get("density_gt_semantics"))
        model_config = source_checkpoint["config"]
        args = validate_config(argparse.Namespace(**apply_checkpoint_semantics(vars(args), model_config)))
    save_config(args)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    student = build_formal_model_from_config(model_config).to(device)
    if source_checkpoint is not None:
        student.load_state_dict(source_checkpoint["model"], strict=True)
    teacher = initialize_teacher(student)
    ema_checkpoint_config = dict(model_config)
    ema_config_keys = (
        "ema_decay", "ema_sigma_j", "ema_sigma_m", "ema_sigma_r",
        "ema_stability_min_weight", "lambda_ema_j", "lambda_ema_m", "lambda_ema_r",
        "lambda_anchor", "real_batch_size", "source_anchor_batch_size", "real_data_dir",
        "learning_rate", "epochs", "num_workers", "source_checkpoint",
    )
    ema_checkpoint_config.update({key: getattr(args, key) for key in ema_config_keys})
    optimizer = AdamW(student.parameters(), lr=args.learning_rate)
    geometry_generator = torch.Generator(device=device)
    geometry_generator.manual_seed(args.model_init_seed + 201)
    omega_generator = torch.Generator(device=device)
    omega_generator.manual_seed(args.model_init_seed + 101)
    # Formal real data carries no GT/masks.  EMA keeps a separate source
    # iterator for the full anchor objective; construction is explicit here.
    real_dataset = RealMultiModalDataset(
        f"{args.real_data_dir}/hazy", f"{args.real_data_dir}/tir",
        pair_alignment_policy=args.pair_alignment_policy,
        tir_normalization_config=tir_normalization_config_from_args(args),
    )
    source_dataset = SynthMultiModalDataset(
        args.train_data_dir, train=True, size=args.train_size, density_gt_semantics=args.density_gt_semantics,
        density_map_normalization=args.density_map_normalization,
        density_fixed_min=args.density_fixed_min, density_fixed_max=args.density_fixed_max,
        density_calibrated_min=args.density_calibrated_min,
        density_calibrated_max=args.density_calibrated_max,
        tir_normalization_config=tir_normalization_config_from_args(args),
        pair_alignment_policy=args.pair_alignment_policy,
        augmentation_seed_base=args.model_init_seed,
    )
    inspection = source_dataset.inspect_density_sample(0)
    print(f"[density][EMA anchor] semantics={args.density_gt_semantics} inspection={inspection}")
    real_sampler = StatefulRandomSampler(len(real_dataset), seed=args.model_init_seed + 1)
    source_sampler = StatefulRandomSampler(len(source_dataset), seed=args.model_init_seed + 2)
    source_dataset.set_sampler_epoch(source_sampler.epoch)
    real_loader = DataLoader(real_dataset, batch_size=args.real_batch_size, sampler=real_sampler,
                             num_workers=args.num_workers, collate_fn=collate_real)
    source_loader = DataLoader(source_dataset, batch_size=args.source_anchor_batch_size, sampler=source_sampler,
                               num_workers=args.num_workers, collate_fn=collate_synth)
    student.train()
    set_batchnorm_eval(student)
    set_batchnorm_eval(teacher)
    source_global_step = int(source_checkpoint.get("global_step", 0)) if source_checkpoint is not None else 0
    ema_global_step, start_epoch = 0, 0
    source_empty_omega_streak, anchor_empty_omega_streak = 0, 0
    if resume_checkpoint is not None:
        restored = restore_ema_training_state(
            resume_checkpoint, student, teacher, optimizer, source_sampler, real_sampler,
            args.density_gt_semantics, manifest_fingerprints={
                "source": source_dataset.manifest_fingerprint(),
                "real": real_dataset.manifest_fingerprint(),
            }, omega_generator=omega_generator, geometry_generator=geometry_generator,
        )
        source_global_step = restored["source_global_step"]
        ema_global_step = restored["ema_global_step"]
        start_epoch = restored["epoch"]
        source_dataset.set_sampler_epoch(source_sampler.epoch)
        source_empty_omega_streak = restored["source_empty_omega_streak"]
        anchor_empty_omega_streak = restored["anchor_empty_omega_streak"]
        # A checkpoint is written at a successful-step boundary.  A checkpoint
        # taken at the end of an epoch therefore carries exhausted cursors;
        # move each independent stream to its next deterministic permutation
        # before constructing the next epoch's iterators.
        if start_epoch < args.epochs and real_sampler.next_sample_position >= len(real_dataset):
            real_sampler.advance_epoch()
        if start_epoch < args.epochs and source_sampler.next_sample_position >= len(source_dataset):
            source_sampler.advance_epoch()
            source_dataset.set_sampler_epoch(source_sampler.epoch)
    for _epoch in range(start_epoch, args.epochs):
        source_global_step, ema_global_step, anchor_empty_omega_streak = run_ema_epoch(
            student, teacher, optimizer, real_loader, source_loader, args,
            source_global_step, ema_global_step, real_sampler, source_sampler, anchor_empty_omega_streak,
            geometry_generator=geometry_generator, omega_generator=omega_generator,
        )
        if real_sampler.next_sample_position >= len(real_dataset) and _epoch + 1 < args.epochs:
            real_sampler.advance_epoch()
        if source_sampler.next_sample_position >= len(source_dataset) and _epoch + 1 < args.epochs:
            source_sampler.advance_epoch()
            source_dataset.set_sampler_epoch(source_sampler.epoch)
        if args.saved_model_dir:
            checkpoint = build_ema_checkpoint(
                student.state_dict(), teacher.state_dict(), optimizer.state_dict(), None,
                source_global_step, ema_global_step, _epoch + 1, ema_checkpoint_config,
                args.density_gt_semantics,
                capture_rng_state(omega_generator=omega_generator, geometry_generator=geometry_generator),
                sampler_states={"source": source_sampler.state_dict(), "real": real_sampler.state_dict()},
                empty_omega_streaks={"source": source_empty_omega_streak, "anchor": anchor_empty_omega_streak},
                manifest_fingerprints={
                    "source": source_dataset.manifest_fingerprint(),
                    "real": real_dataset.manifest_fingerprint(),
                },
            )
            torch.save(checkpoint, Path(args.saved_model_dir) / "ema_last.pt")
    return student, teacher


if __name__ == "__main__":
    main()
