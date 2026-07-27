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
from option.EMA import (
    build_ema_checkpoint_config, build_parser, prepare_experiment_dirs,
    resolve_ema_config, save_config, tir_normalization_config_from_args,
    validate_config,
)
from training.paired_geometry import sample_geometry
from loss.real.consistency import real_consistency_loss, stability_weights
from training.checkpointing import (
    build_ema_checkpoint, build_formal_model_from_config, capture_rng_state,
    restore_ema_training_state, validate_checkpoint_metadata,
)
from training.ema_step import run_ema_adaptation_step
from training.step_transaction import rollback_step_transaction, snapshot_step_transaction
from training.omega_sampler import OmegaSampler
from training.omega_state import update_empty_omega_streak
from training.source_step import compute_source_batch_losses


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


def _log_loss_components(args):
    print("q components:", f"l1={args.q_l1_weight}", f"gradient={args.q_gradient_weight}", f"ssim={args.q_ssim_weight}")
    print("reconstruction components:", f"l1={args.rec_l1_weight}", f"gradient={args.rec_gradient_weight}", f"ssim={args.rec_ssim_weight}")
    print("boundary components:", f"l1={args.boundary_l1_weight}", f"gradient={args.boundary_gradient_weight}")
    if args.q_gradient_weight == 0 and args.q_ssim_weight == 0:
        print("WARNING: q supervision is L1-only")
    if args.rec_gradient_weight == 0 and args.rec_ssim_weight == 0:
        print("WARNING: reconstruction is L1-only")
    if args.boundary_gradient_weight == 0:
        print("WARNING: boundary loss has no gradient component")


def run_ema_views(teacher, student, hazy_rgb, tir, generator, route_temperature,
                  sigma_j, sigma_m, sigma_r, minimum_weight,
                  lambda_j=1.0, lambda_m=1.0, lambda_r=1.0):
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
                                   r_s, 0.5 * (r_a + r_b), *weights,
                                   lambda_j=lambda_j, lambda_m=lambda_m, lambda_r=lambda_r)
    return losses, (transform_a, transform_b, transform_s)


def run_ema_epoch(student, teacher, optimizer, real_loader, source_loader, args,
                   source_global_step=0, ema_global_step=0, real_sampler=None, source_sampler=None,
                  anchor_empty_omega_streak=0, geometry_generator=None, omega_generator=None,
                  real_loader_generator=None, source_loader_generator=None,
                  ema_failed_step_streak=0):
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
    if real_loader_generator is None:
        real_loader_generator = torch.Generator().manual_seed(args.model_init_seed + 302)
    if source_loader_generator is None:
        source_loader_generator = torch.Generator().manual_seed(args.model_init_seed + 303)
    omega_sampler = OmegaSampler(args.omega_regions_per_image, args.omega_min_area,
                                 args.omega_max_area, seed=args.model_init_seed)
    student.train()
    set_batchnorm_eval(student)
    teacher.eval()
    set_batchnorm_eval(teacher)
    real_iterator_state_before = real_loader_generator.get_state()
    real_iterator = iter(real_loader)
    source_iterator_state_before = source_loader_generator.get_state()
    source_iterator = iter(source_loader)
    step_failed = False
    for real_batch in real_iterator:
        try:
            source_batch = next(source_iterator)
        except StopIteration:
            if source_sampler is not None and source_sampler.next_sample_position >= source_sampler.data_source_size:
                source_sampler.advance_epoch()
                source_loader.dataset.set_sampler_epoch(source_sampler.epoch)
            source_iterator_state_before = source_loader_generator.get_state()
            source_iterator = iter(source_loader)
            source_batch = next(source_iterator)
        real_hazy, real_tir, _metadata = real_batch
        real_hazy, real_tir = real_hazy.to(device), real_tir.to(device)
        source_batch = tuple(value.to(device) for value in source_batch)
        transaction = snapshot_step_transaction(
            modules=[student, teacher], omega_generator=omega_generator,
            geometry_generator=geometry_generator,
            dataloader_generators={
                "real": real_loader_generator,
                "source_anchor": source_loader_generator,
            },
        )

        def real_loss_fn():
            losses, _ = run_ema_views(
                teacher, student, real_hazy, real_tir, geometry_generator, args.route_tau_end,
                args.ema_sigma_j, args.ema_sigma_m, args.ema_sigma_r, args.ema_stability_min_weight,
                args.lambda_ema_j, args.lambda_ema_m, args.lambda_ema_r,
            )
            return losses["L_real"]

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
            rollback_step_transaction(
                transaction, modules=[student, teacher], omega_generator=omega_generator,
                geometry_generator=geometry_generator,
                dataloader_generators={
                    "real": real_loader_generator,
                    "source_anchor": source_loader_generator,
                },
            )
            optimizer.zero_grad(set_to_none=True)
            real_loader_generator.set_state(real_iterator_state_before)
            source_loader_generator.set_state(source_iterator_state_before)
            ema_failed_step_streak += 1
            if ema_failed_step_streak >= args.max_consecutive_failed_steps:
                raise RuntimeError(
                    "EMA optimizer failed repeatedly; "
                    f"ema_step={ema_global_step}, failed_streak={ema_failed_step_streak}, "
                    f"loss={float(result['L_adapt'])}, nonfinite_gradients=True, "
                    f"real_sampler_position={real_sampler.next_sample_position if real_sampler else None}, "
                    f"source_sampler_position={source_sampler.next_sample_position if source_sampler else None}, "
                    f"real_batch_size={real_hazy.shape[0]}, source_batch_size={source_batch[0].shape[0]}, "
                    "grad_scaler_scale=None"
                )
            print(f"ema_step={ema_global_step} skipped: transaction rolled back; pair will be retried")
            step_failed = True
            break
        if result["step_succeeded"]:
            ema_failed_step_streak = 0
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
                valid_omega_count=int(source_result["route_supervision"]["valid_q_region_count"]),
                step_succeeded=True,
            )
            if anchor_empty_omega_streak >= args.max_consecutive_empty_omega_steps:
                raise RuntimeError(
                    "EMA source anchor route supervision remained unavailable; "
                    f"streak={anchor_empty_omega_streak}, statuses={source_result['omega']['status']}, "
                    f"sampled_omega_count={source_result['route_supervision']['sampled_omega_count']}, "
                    f"valid_q_region_count={source_result['route_supervision']['valid_q_region_count']}, "
                    f"valid_route_pixel_count={source_result['route_supervision']['valid_route_pixel_count']}, "
                    f"effective_lambda_route={args.lambda_router * source_result['state']['lambda_route']:.6f}"
                )
            if ema_global_step % 50 == 0:
                print(
                    f"ema_step={ema_global_step} source_step={source_global_step} "
                    f"L_real={result['L_real'].item():.5f} L_src={result['L_src'].item():.5f} "
                    f"anchor={args.lambda_anchor * result['L_src'].item():.5f}"
                )
    return {
        "epoch_completed": bool(real_sampler is not None and real_sampler.next_sample_position >= real_sampler.data_source_size),
        "step_failed": step_failed,
        "source_global_step": source_global_step,
        "ema_global_step": ema_global_step,
        "anchor_empty_omega_streak": anchor_empty_omega_streak,
        "ema_failed_step_streak": ema_failed_step_streak,
    }


def main(argv=None):
    raw_args = build_parser().parse_args(argv)
    resume_checkpoint = torch.load(raw_args.resume_checkpoint, map_location="cpu") if raw_args.resume_checkpoint else None
    if resume_checkpoint is None and not raw_args.source_checkpoint:
        raise ValueError("EMA adaptation requires --source_checkpoint or --resume_checkpoint")
    source_checkpoint = None
    if resume_checkpoint is not None:
        validate_checkpoint_metadata(resume_checkpoint, "ema", resume_checkpoint.get("density_gt_semantics"))
        model_config = resume_checkpoint["config"]
        resolved, config_diff = resolve_ema_config(
            raw_args, model_config,
            allow_training_override=raw_args.allow_ema_training_override,
            is_resume=True,
        )
        args = validate_config(argparse.Namespace(**resolved))
        if config_diff:
            print(f"[EMA] explicit training override: {config_diff}")
    else:
        source_checkpoint = torch.load(raw_args.source_checkpoint, map_location="cpu")
        validate_checkpoint_metadata(source_checkpoint, "source", source_checkpoint.get("density_gt_semantics"))
        model_config = source_checkpoint["config"]
        resolved, _ = resolve_ema_config(
            raw_args, model_config, allow_training_override=False,
        )
        args = validate_config(argparse.Namespace(**resolved))
    _log_loss_components(args)
    validate_single_process_world()
    prepare_experiment_dirs(args)
    random.seed(args.model_init_seed)
    np.random.seed(args.model_init_seed)
    torch.manual_seed(args.model_init_seed)
    save_config(args)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    student = build_formal_model_from_config(model_config).to(device)
    if source_checkpoint is not None:
        student.load_state_dict(source_checkpoint["model"], strict=True)
    teacher = initialize_teacher(student)
    ema_checkpoint_config = build_ema_checkpoint_config(model_config, args)
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
        args.source_anchor_data_dir, train=True, size=args.train_size, density_gt_semantics=args.density_gt_semantics,
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
    real_loader_generator = torch.Generator().manual_seed(args.model_init_seed + 302)
    source_anchor_loader_generator = torch.Generator().manual_seed(args.model_init_seed + 303)
    real_loader = DataLoader(real_dataset, batch_size=args.real_batch_size, sampler=real_sampler,
                              num_workers=args.num_workers, collate_fn=collate_real,
                              generator=real_loader_generator)
    source_loader = DataLoader(source_dataset, batch_size=args.source_anchor_batch_size, sampler=source_sampler,
                                num_workers=args.num_workers, collate_fn=collate_synth,
                                generator=source_anchor_loader_generator)
    student.train()
    set_batchnorm_eval(student)
    set_batchnorm_eval(teacher)
    source_global_step = int(source_checkpoint.get("global_step", 0)) if source_checkpoint is not None else 0
    ema_global_step, start_epoch = 0, 0
    source_empty_omega_streak, anchor_empty_omega_streak = 0, 0
    ema_failed_step_streak = 0
    if resume_checkpoint is not None:
        restored = restore_ema_training_state(
            resume_checkpoint, student, teacher, optimizer, source_sampler, real_sampler,
            args.density_gt_semantics, manifest_fingerprints={
                "source": source_dataset.manifest_fingerprint(),
                "real": real_dataset.manifest_fingerprint(),
            }, omega_generator=omega_generator, geometry_generator=geometry_generator,
            dataloader_generators={
                "real": real_loader_generator,
                "source_anchor": source_anchor_loader_generator,
            },
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
    epoch = start_epoch
    while epoch < args.epochs:
        epoch_result = run_ema_epoch(
            student, teacher, optimizer, real_loader, source_loader, args,
            source_global_step, ema_global_step, real_sampler, source_sampler, anchor_empty_omega_streak,
            geometry_generator=geometry_generator, omega_generator=omega_generator,
            real_loader_generator=real_loader_generator,
            source_loader_generator=source_anchor_loader_generator,
            ema_failed_step_streak=ema_failed_step_streak,
        )
        source_global_step = epoch_result["source_global_step"]
        ema_global_step = epoch_result["ema_global_step"]
        anchor_empty_omega_streak = epoch_result["anchor_empty_omega_streak"]
        ema_failed_step_streak = epoch_result["ema_failed_step_streak"]
        if epoch_result["step_failed"]:
            continue
        if not epoch_result["epoch_completed"]:
            continue
        epoch += 1
        if args.saved_model_dir:
            checkpoint = build_ema_checkpoint(
                student.state_dict(), teacher.state_dict(), optimizer.state_dict(), None,
                source_global_step, ema_global_step, epoch, ema_checkpoint_config,
                args.density_gt_semantics,
                capture_rng_state(
                    omega_generator=omega_generator, geometry_generator=geometry_generator,
                    dataloader_generators={
                        "real": real_loader_generator,
                        "source_anchor": source_anchor_loader_generator,
                    },
                ),
                sampler_states={"source": source_sampler.state_dict(), "real": real_sampler.state_dict()},
                empty_omega_streaks={"source": source_empty_omega_streak, "anchor": anchor_empty_omega_streak},
                manifest_fingerprints={
                    "source": source_dataset.manifest_fingerprint(),
                    "real": real_dataset.manifest_fingerprint(),
                },
            )
            torch.save(checkpoint, Path(args.saved_model_dir) / "ema_last.pt")
        if epoch < args.epochs:
            real_sampler.advance_epoch()
            if source_sampler.next_sample_position >= len(source_dataset):
                source_sampler.advance_epoch()
                source_dataset.set_sampler_epoch(source_sampler.epoch)
    return student, teacher


if __name__ == "__main__":
    main()
