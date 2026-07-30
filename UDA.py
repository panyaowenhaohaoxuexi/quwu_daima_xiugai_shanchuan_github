"""Two-stage FLIR-to-M3FD unsupervised domain adaptation entry point.

Stage A uses unlabelled M3FD only as a bounded RGB/TIR statistics reference
while retaining the complete FLIR physical five-tuple supervision.  Stage B
then runs the existing EMA adaptation with that Stage-A model as its Source
anchor and delays target-domain route self-consistency.
"""

import argparse
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

from data.data_loader import (RealMultiModalDataset, SynthMultiModalDataset, collate_real, collate_synth,
                              load_tir_as_float_tensor)
from loss.source import build_regional_reconstruction_criteria
from option.EMA import (prepare_experiment_dirs, real_modal_dirs_from_args, save_config,
                        tir_normalization_config_from_args)
from option.UDA import (build_parser, build_uda_checkpoint_config, resolve_uda_config,
                        validate_config)
from training.real_adaptation import real_adaptation_loss
from training.schedule import build_coa_adam, cycle_batches, set_cosine_learning_rate
from training.source import compute_physical_mask_batch_losses
from training.target_style import apply_target_statistics
from training.validation import evaluate_paired_validation, save_best_if_improved
from training.observability import TrainingLogger
from utils.checkpoint import (build_ema_checkpoint, build_model_from_config, build_source_checkpoint,
                              load_ema_checkpoint, load_strict_v2_state_dict,
                              preflight_ema_resume_checkpoint, preflight_source_initialization_checkpoint)
from utils.metrics import psnr, ssim_global


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _device_from_args(args):
    return torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")


def _require_finite_loss(loss, model, *, stage, epoch, step):
    if not torch.isfinite(loss).all():
        raise RuntimeError(f"{stage} non-finite loss: epoch={epoch} step={step} loss={float(loss.detach())}")
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise RuntimeError(
                f"{stage} non-finite gradient: epoch={epoch} step={step} "
                f"loss={float(loss.detach())} parameter={name}"
            )


def _set_batchnorm_eval(module):
    for child in module.modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            child.eval()


def _tensor_to_rgb_image(value):
    value = value.detach().float().cpu()[0].clamp(0.0, 1.0)
    if value.shape[0] == 1:
        value = value.repeat(3, 1, 1)
    if value.shape[0] != 3:
        raise ValueError("probe tensors must have one or three channels")
    return Image.fromarray(value.permute(1, 2, 0).mul(255).round().byte().numpy(), mode="RGB")


@torch.inference_mode()
def save_target_probe(model, hazy, tir, output_dir, *, route_temperature, prefix):
    """Save target-only dehazing and routing diagnostics without target labels."""
    was_training = model.training
    model.eval()
    output = model(hazy, tir, route_temperature=route_temperature, route_mode="hard")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("pred_clear", "density_map", "route_soft", "route_hard", "boundary_map"):
        _tensor_to_rgb_image(output[name]).save(output_dir / f"{prefix}_{name}.png")
    if was_training:
        model.train()
    return {"density_mean": float(output["density_map"].mean().cpu()),
            "route_hard_fraction": float(output["route_hard"].mean().cpu())}


def _load_target_probe(args, device):
    if not args.probe_hazy:
        return None
    hazy = TF.to_tensor(Image.open(args.probe_hazy).convert("RGB")).unsqueeze(0)
    tir = load_tir_as_float_tensor(args.probe_tir, tir_normalization_config_from_args(args)).unsqueeze(0)
    if tir.shape[-2:] != hazy.shape[-2:]:
        if args.pair_alignment_policy == "strict":
            raise ValueError("probe RGB/TIR alignment mismatch under pair_alignment_policy='strict'")
        tir = F.interpolate(tir, size=hazy.shape[-2:], mode="bilinear", align_corners=False)
    return hazy.to(device), tir.to(device)


def _write_probe_if_requested(model, args, probe, *, prefix):
    if probe is None:
        return None
    return save_target_probe(
        model, *probe, args.probe_output_dir, route_temperature=args.route_tau_end, prefix=prefix,
    )


def route_consistency_multiplier(step, *, warmup_steps, ramp_steps):
    """Keep route self-consistency off before its target-domain warm-up ends."""
    if step < 0 or warmup_steps < 0 or ramp_steps < 0:
        raise ValueError("route consistency steps must be non-negative")
    if step <= warmup_steps:
        return 0.0
    if ramp_steps == 0:
        return 1.0
    return min(1.0, (step - warmup_steps) / float(ramp_steps))


def style_source_batch(source_batch, target_hazy, target_tir, *, generator, probability,
                       beta_min, beta_max, min_gain, max_gain, max_abs_bias):
    """Apply target statistics only to Source modalities, never to physical labels."""
    hazy, clear, tir, density_gt, completion_mask_gt = source_batch
    if target_hazy.shape[0] != hazy.shape[0] or target_tir.shape[0] != hazy.shape[0]:
        raise ValueError("target reference batch size must match the Source batch")
    if not 0.0 <= float(probability) <= 1.0 or not 0.0 <= float(beta_min) <= float(beta_max) <= 1.0:
        raise ValueError("invalid target style probability or beta range")
    apply_mask = torch.rand(hazy.shape[0], generator=generator, device="cpu") < float(probability)
    beta = torch.empty(hazy.shape[0], device=hazy.device, dtype=hazy.dtype).uniform_(
        float(beta_min), float(beta_max), generator=generator
    )
    beta = beta * apply_mask.to(device=hazy.device, dtype=hazy.dtype)
    styled = apply_target_statistics(
        hazy, clear, tir, target_hazy.to(device=hazy.device, dtype=hazy.dtype),
        target_tir.to(device=hazy.device, dtype=hazy.dtype), beta=beta,
        min_gain=min_gain, max_gain=max_gain, max_abs_bias=max_abs_bias,
    )
    return (styled.hazy, styled.clear, styled.tir, density_gt, completion_mask_gt), apply_mask.to(hazy.device)


def _build_datasets_and_loaders(args, *, target_batch_size):
    """Construct only formal paired datasets; M3FD has no invented labels."""
    normalizer = tir_normalization_config_from_args(args)
    real_hazy_dir, real_tir_dir = real_modal_dirs_from_args(args)
    real_dataset = RealMultiModalDataset(
        real_hazy_dir, real_tir_dir, pair_alignment_policy=args.pair_alignment_policy,
        tir_normalization_config=normalizer,
    )
    source_dataset = SynthMultiModalDataset(
        args.source_anchor_data_dir, train=True, size=args.train_size,
        density_gt_semantics=args.density_gt_semantics, density_map_normalization=args.density_map_normalization,
        density_fixed_min=args.density_fixed_min, density_fixed_max=args.density_fixed_max,
        density_calibrated_min=args.density_calibrated_min, density_calibrated_max=args.density_calibrated_max,
        tir_normalization_config=normalizer, pair_alignment_policy=args.pair_alignment_policy,
        augmentation_seed_base=args.model_init_seed,
    )
    validation_dataset = SynthMultiModalDataset(
        args.validation_data_dir, train=False, size="full", density_gt_semantics=args.density_gt_semantics,
        density_map_normalization=args.density_map_normalization, density_fixed_min=args.density_fixed_min,
        density_fixed_max=args.density_fixed_max, density_calibrated_min=args.density_calibrated_min,
        density_calibrated_max=args.density_calibrated_max, tir_normalization_config=normalizer,
        pair_alignment_policy=args.pair_alignment_policy,
    )
    return (
        real_dataset, source_dataset, validation_dataset,
        DataLoader(real_dataset, batch_size=target_batch_size, shuffle=True, num_workers=args.num_workers,
                   collate_fn=collate_real),
        DataLoader(source_dataset, batch_size=args.source_anchor_batch_size, shuffle=True,
                   num_workers=args.num_workers, collate_fn=collate_synth),
        DataLoader(validation_dataset, batch_size=args.validation_batch_size, shuffle=False,
                   num_workers=args.num_workers, collate_fn=collate_synth),
    )


def _stage_summary(stage, args, device, datasets, model, *, initialization_checkpoint):
    real_dataset, source_dataset, validation_dataset = datasets
    return {
        "stage": stage, "device": str(device), "seed": args.model_init_seed,
        "dataset_sizes": {"real": len(real_dataset), "source_anchor": len(source_dataset),
                          "validation": len(validation_dataset)},
        "batch_configuration": {"real": args.real_batch_size, "source_anchor": args.source_anchor_batch_size,
                                "validation": args.validation_batch_size, "num_workers": args.num_workers},
        "schedule": {"epochs": args.epochs, "iters_per_epoch": args.iters_per_epoch,
                     "total_steps": args.epochs * args.iters_per_epoch, "start_lr": args.start_lr,
                     "end_lr": args.end_lr},
        "parameter_counts": {"total": sum(parameter.numel() for parameter in model.parameters()),
                             "rgb_encoder": sum(parameter.numel() for parameter in model.rgb_encoder.parameters()),
                             "tir_encoder": sum(parameter.numel() for parameter in model.tir_encoder.parameters())},
        "res2net_pretrained_loaded": True, "res2net_pretrained_path": model.rgb_encoder.pretrained_path,
        "initialization_checkpoint": initialization_checkpoint,
        "output_directories": {"experiment": args.exp_dir, "models": args.saved_model_dir,
                               "diagnostics": args.saved_data_dir},
    }


def _run_source_style_stage(args, checkpoint, checkpoint_config):
    """Fine-tune a Source model using bounded M3FD statistics references."""
    device = _device_from_args(args)
    reconstruction_criteria = build_regional_reconstruction_criteria(device)
    model = build_model_from_config(vars(args)).to(device)
    source_preflight = preflight_source_initialization_checkpoint(checkpoint)
    load_strict_v2_state_dict(model, source_preflight["states"]["model"], label="Source model")
    optimizer = build_coa_adam(model.parameters(), learning_rate=args.start_lr)
    datasets_and_loaders = _build_datasets_and_loaders(args, target_batch_size=args.source_anchor_batch_size)
    real_dataset, source_dataset, validation_dataset, real_loader, source_loader, validation_loader = datasets_and_loaders
    prepare_experiment_dirs(args)
    save_config(args)
    logger = TrainingLogger(args.exp_dir or args.saved_model_dir)
    logger.write_run_summary(_stage_summary(
        "source_style", args, device, (real_dataset, source_dataset, validation_dataset), model,
        initialization_checkpoint=args.source_checkpoint,
    ))
    total_steps = args.epochs * args.iters_per_epoch
    global_step = source_preflight["metadata"]["global_step"]
    best_psnr = float("-inf")
    probe = _load_target_probe(args, device)
    initial_probe = _write_probe_if_requested(model, args, probe, prefix="source_style_initial")
    if initial_probe is not None:
        logger.log_event("target_probe", global_step=global_step, probe_phase="initial", **initial_probe)
    source_iterator, real_iterator = cycle_batches(source_loader), cycle_batches(real_loader)
    style_generator = torch.Generator().manual_seed(args.model_init_seed + 101)
    print(f"uda source_style startup device={device} source={len(source_dataset)} target_reference={len(real_dataset)} "
          f"validation={len(validation_dataset)} total_steps={total_steps}")
    for epoch in range(args.epochs):
        source_dataset.set_sampler_epoch(epoch)
        model.train()
        for logical_step in range(args.iters_per_epoch):
            started = perf_counter()
            target_hazy, target_tir, _ = next(real_iterator)
            source_batch = next(source_iterator)
            if source_batch[0].numel() == 0:
                continue
            styled_source, styled_samples = style_source_batch(
                source_batch, target_hazy, target_tir, generator=style_generator,
                probability=args.style_probability, beta_min=args.style_beta_min,
                beta_max=args.style_beta_max, min_gain=args.style_min_gain,
                max_gain=args.style_max_gain, max_abs_bias=args.style_max_abs_bias,
            )
            schedule_step = epoch * args.iters_per_epoch + logical_step + 1
            learning_rate = set_cosine_learning_rate(
                optimizer, step=schedule_step, total_steps=total_steps, start_lr=args.start_lr,
                end_lr=args.end_lr, no_lr_sche=args.no_lr_sche,
            )
            styled_source = tuple(value.to(device) for value in styled_source)
            optimizer.zero_grad(set_to_none=True)
            result = compute_physical_mask_batch_losses(
                model, styled_source, args, global_step, reconstruction_criteria=reconstruction_criteria,
            )
            loss = result["losses"]["total"]
            _require_finite_loss(loss, model, stage="uda source_style", epoch=epoch, step=global_step)
            optimizer.step()
            global_step += 1
            hazy, clear = styled_source[0], styled_source[1]
            logger.log_event(
                "train_step", epoch=epoch + 1, epoch_step=logical_step + 1, global_step=global_step,
                learning_rate=learning_rate, duration_seconds=perf_counter() - started,
                styled_fraction=styled_samples.float().mean(), train_psnr=psnr(result["output"]["pred_clear"].detach(), clear),
                train_ssim=ssim_global(result["output"]["pred_clear"].detach(), clear),
                **{f"loss_{name}": value for name, value in result["losses"].items()},
                **{f"schedule_{name}": value for name, value in result["state"].items()},
            )
        validation = evaluate_paired_validation(model, validation_loader, device, route_temperature=args.route_tau_end)
        candidate = build_source_checkpoint(
            model, optimizer, epoch=epoch + 1, global_step=global_step,
            config=build_uda_checkpoint_config(checkpoint_config, args), best_psnr=max(best_psnr, validation["psnr"]),
        )
        is_best = validation["psnr"] > best_psnr
        if args.saved_model_dir:
            output_dir = Path(args.saved_model_dir)
            best_psnr = save_best_if_improved(validation["psnr"], best_psnr, candidate, output_dir / "source_style_best.pt")
            candidate["best_psnr"] = best_psnr
            torch.save(candidate, output_dir / "source_style_last.pt")
        logger.log_event("validation", epoch=epoch + 1, global_step=global_step, psnr=validation["psnr"],
                         ssim=validation["ssim"], best_psnr=best_psnr, is_best=is_best)
        probe_metrics = _write_probe_if_requested(model, args, probe, prefix=f"source_style_epoch_{epoch + 1:04d}")
        if probe_metrics is not None:
            logger.log_event("target_probe", epoch=epoch + 1, global_step=global_step,
                             probe_phase="validation", **probe_metrics)
        print(f"uda source_style validation epoch={epoch + 1} psnr={validation['psnr']:.4f} "
              f"ssim={validation['ssim']:.4f} best_psnr={best_psnr:.4f}")
    return model


def _run_ema_stage(args, checkpoint, checkpoint_config, *, resume):
    """Run Stage-B real EMA adaptation with delayed route self-consistency."""
    from EMA import (adaptation_loss, initialize_coa_clip, initialize_teacher, require_coa_clip_cuda,
                     update_teacher_after_success)

    device = _device_from_args(args)
    reconstruction_criteria = build_regional_reconstruction_criteria(device)
    require_coa_clip_cuda(device)
    clip_criterion, text_features = initialize_coa_clip(device)
    student = build_model_from_config(vars(args)).to(device)
    optimizer = build_coa_adam(student.parameters(), learning_rate=args.start_lr)
    if resume:
        restored_preflight = preflight_ema_resume_checkpoint(checkpoint)
        teacher = initialize_teacher(student)
        restored = load_ema_checkpoint(checkpoint, student, teacher, optimizer)
        start_epoch, source_global_step, ema_global_step = (
            restored["epoch"], restored["source_global_step"], restored["ema_global_step"]
        )
        best_psnr = restored.get("best_psnr", float("-inf"))
    else:
        source_preflight = preflight_source_initialization_checkpoint(checkpoint)
        load_strict_v2_state_dict(student, source_preflight["states"]["model"], label="Source model")
        teacher = initialize_teacher(student)
        start_epoch, source_global_step, ema_global_step, best_psnr = (
            0, source_preflight["metadata"]["global_step"], 0, float("-inf")
        )
    datasets_and_loaders = _build_datasets_and_loaders(args, target_batch_size=args.real_batch_size)
    real_dataset, source_dataset, validation_dataset, real_loader, source_loader, validation_loader = datasets_and_loaders
    prepare_experiment_dirs(args)
    save_config(args)
    logger = TrainingLogger(args.exp_dir or args.saved_model_dir, resume=resume)
    summary = _stage_summary("ema", args, device, (real_dataset, source_dataset, validation_dataset), student,
                             initialization_checkpoint=args.resume_checkpoint or args.source_checkpoint)
    summary["route_consistency_schedule"] = {"warmup_steps": args.route_consistency_warmup_steps,
                                             "ramp_steps": args.route_consistency_ramp_steps}
    logger.write_run_summary(summary)
    total_steps = args.epochs * args.iters_per_epoch
    real_iterator, source_iterator = cycle_batches(real_loader), cycle_batches(source_loader)
    geometry_generator = torch.Generator().manual_seed(args.model_init_seed + 201)
    probe = _load_target_probe(args, device)
    initial_probe = _write_probe_if_requested(student, args, probe, prefix="ema_initial")
    if initial_probe is not None:
        logger.log_event("target_probe", ema_global_step=ema_global_step, probe_phase="initial", **initial_probe)
    print(f"uda ema startup device={device} real={len(real_dataset)} source_anchor={len(source_dataset)} "
          f"validation={len(validation_dataset)} total_steps={total_steps}")
    for epoch in range(start_epoch, args.epochs):
        source_dataset.set_sampler_epoch(epoch)
        student.train(); _set_batchnorm_eval(student); teacher.eval(); _set_batchnorm_eval(teacher)
        for logical_step in range(args.iters_per_epoch):
            started = perf_counter()
            real_hazy, real_tir, _ = next(real_iterator)
            source_batch = next(source_iterator)
            schedule_step = epoch * args.iters_per_epoch + logical_step + 1
            learning_rate = set_cosine_learning_rate(
                optimizer, step=schedule_step, total_steps=total_steps, start_lr=args.start_lr,
                end_lr=args.end_lr, no_lr_sche=args.no_lr_sche,
            )
            route_multiplier = route_consistency_multiplier(
                ema_global_step, warmup_steps=args.route_consistency_warmup_steps,
                ramp_steps=args.route_consistency_ramp_steps,
            )
            optimizer.zero_grad(set_to_none=True)
            real = real_adaptation_loss(
                teacher, student, real_hazy.to(device), real_tir.to(device), geometry_generator, args,
                route_multiplier=route_multiplier, clip_criterion=clip_criterion, text_features=text_features,
            )
            source = compute_physical_mask_batch_losses(
                student, tuple(value.to(device) for value in source_batch), args, source_global_step,
                reconstruction_criteria=reconstruction_criteria,
            )
            adapt = adaptation_loss(real, source, args)
            _require_finite_loss(adapt, student, stage="uda ema", epoch=epoch, step=ema_global_step)
            optimizer.step()
            source_global_step += 1; ema_global_step += 1
            logger.log_event(
                "train_step", epoch=epoch + 1, epoch_step=logical_step + 1,
                source_global_step=source_global_step, ema_global_step=ema_global_step,
                learning_rate=learning_rate, duration_seconds=perf_counter() - started,
                route_consistency_multiplier=route_multiplier, loss_total=adapt,
                loss_real=real["L_real"], loss_clip=real["L_clip"], loss_source_anchor=source["losses"]["total"],
                **{f"loss_real_{name.lower()}": value for name, value in real.items()},
                **{f"loss_source_{name}": value for name, value in source["losses"].items()},
                **{f"schedule_{name}": value for name, value in source["state"].items()},
            )
        update_teacher_after_success(teacher, student, args.ema_decay)
        validation = evaluate_paired_validation(student, validation_loader, device, route_temperature=args.route_tau_end)
        candidate = build_ema_checkpoint(
            student, teacher, optimizer, epoch=epoch + 1, source_global_step=source_global_step,
            ema_global_step=ema_global_step, config=build_uda_checkpoint_config(checkpoint_config, args),
            best_psnr=max(best_psnr, validation["psnr"]),
        )
        is_best = validation["psnr"] > best_psnr
        if args.saved_model_dir:
            output_dir = Path(args.saved_model_dir)
            best_psnr = save_best_if_improved(validation["psnr"], best_psnr, candidate, output_dir / "uda_ema_best.pt")
            candidate["best_psnr"] = best_psnr
            torch.save(candidate, output_dir / "uda_ema_last.pt")
        logger.log_event("validation", epoch=epoch + 1, source_global_step=source_global_step,
                         ema_global_step=ema_global_step, psnr=validation["psnr"], ssim=validation["ssim"],
                         best_psnr=best_psnr, is_best=is_best)
        probe_metrics = _write_probe_if_requested(student, args, probe, prefix=f"ema_epoch_{epoch + 1:04d}")
        if probe_metrics is not None:
            logger.log_event("target_probe", epoch=epoch + 1, ema_global_step=ema_global_step,
                             probe_phase="validation", **probe_metrics)
        print(f"uda ema validation epoch={epoch + 1} psnr={validation['psnr']:.4f} "
              f"ssim={validation['ssim']:.4f} best_psnr={best_psnr:.4f}")
    return student, teacher


def main(argv=None):
    raw_args = build_parser().parse_args(argv)
    if raw_args.stage == "source_style":
        if not raw_args.source_checkpoint or raw_args.resume_checkpoint:
            raise ValueError("source_style requires --source_checkpoint and forbids --resume_checkpoint")
        checkpoint = torch.load(raw_args.source_checkpoint, map_location="cpu")
        checkpoint_preflight = preflight_source_initialization_checkpoint(checkpoint)
        args = validate_config(argparse.Namespace(**resolve_uda_config(raw_args, checkpoint_preflight["config"])))
        _set_seed(args.model_init_seed)
        return _run_source_style_stage(args, checkpoint, checkpoint_preflight["config"])
    if bool(raw_args.source_checkpoint) == bool(raw_args.resume_checkpoint):
        raise ValueError("ema requires exactly one of --source_checkpoint or --resume_checkpoint")
    resume = bool(raw_args.resume_checkpoint)
    checkpoint = torch.load(raw_args.resume_checkpoint or raw_args.source_checkpoint, map_location="cpu")
    checkpoint_preflight = (preflight_ema_resume_checkpoint(checkpoint) if resume
                            else preflight_source_initialization_checkpoint(checkpoint))
    args = validate_config(argparse.Namespace(**resolve_uda_config(
        raw_args, checkpoint_preflight["config"], resume=resume,
    )))
    _set_seed(args.model_init_seed)
    return _run_ema_stage(args, checkpoint, checkpoint_preflight["config"], resume=resume)


if __name__ == "__main__":
    main()
