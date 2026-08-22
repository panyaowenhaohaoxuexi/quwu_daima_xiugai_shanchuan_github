"""Real-domain EMA adaptation entry point for fog-routed RGB--TIR dehazing."""

import argparse
import copy
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from data.data_loader import RealMultiModalDataset, SynthMultiModalDataset, collate_real, collate_synth
from loss.ema import real_consistency_loss, stability_weights
from loss.source import build_regional_reconstruction_criteria
from option.EMA import (build_ema_checkpoint_config, build_parser, prepare_experiment_dirs,
                        real_modal_dirs_from_args, resolve_ema_config, save_config,
                        tir_normalization_config_from_args, validate_config)
from training.source import compute_physical_mask_batch_losses
from training.real_adaptation import real_adaptation_loss
from training.schedule import build_coa_adam, cycle_batches, set_cosine_learning_rate
from training.observability import TrainingLogger
from training.validation import evaluate_paired_validation, save_best_if_improved
from utils.checkpoint import (build_ema_checkpoint, build_model_from_config, load_ema_checkpoint,
                              load_strict_v2_state_dict, preflight_ema_resume_checkpoint,
                              preflight_source_initialization_checkpoint)


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_batchnorm_eval(module):
    for child in module.modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            child.eval()


class TextEncoder(nn.Module):
    """CoA's prompt encoder for the frozen CLIP text tower."""

    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        value = prompts + self.positional_embedding.type(self.dtype)
        value = value.permute(1, 0, 2)
        value = self.transformer(value)
        value = value.permute(1, 0, 2)
        value = self.ln_final(value).type(self.dtype)
        return value[torch.arange(value.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection


def require_coa_clip_cuda(device):
    """CoA's unchanged CLIP package creates CUDA tensors at import time."""
    if device.type != "cuda":
        raise RuntimeError("CoA CLIP EMA requires --device cuda")


def initialize_coa_clip(device):
    """Initialize ViT-B/32, RN101, and haze prompt exactly as CoA EMA does."""
    import clip
    from CLIP import L_clip_from_feature

    clip_model, _ = clip.load("ViT-B/32", device=torch.device("cpu"), download_root="./clip_model/")
    clip_model.to(device)
    for parameter in clip_model.parameters():
        parameter.requires_grad = False
    res_model, _ = clip.load("RN101", device=torch.device("cpu"), download_root="./clip_model/")
    res_model.to(device)
    for parameter in res_model.parameters():
        parameter.requires_grad = False
    data = torch.load("./clip_model/haze_prompt.pth")
    new_state_dict = {}
    for name, value in data.items():
        new_state_dict[name[7:]] = value
    embedding_prompt = nn.Parameter(new_state_dict["embedding_prompt"].to(device), requires_grad=False)
    text_encoder = TextEncoder(clip_model)
    tokenized_prompts = torch.cat([clip.tokenize(prompt) for prompt in [" ".join(["X"] * 16)]])
    text_features = text_encoder(embedding_prompt, tokenized_prompts)
    clip_model.eval()
    res_model.eval()
    return L_clip_from_feature().to(device), text_features


def adaptation_loss(real_losses, source_losses, args):
    return (real_losses["L_real"] + args.w_loss_Clip * real_losses["L_clip"]
            + args.lambda_anchor * source_losses["losses"]["total"])


def initialize_teacher(student):
    teacher = copy.deepcopy(student)
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


@torch.no_grad()
def update_teacher_after_success(teacher, student, decay):
    if not 0 <= decay < 1:
        raise ValueError("ema_decay must be in [0, 1)")
    for teacher_parameter, student_parameter in zip(teacher.parameters(), student.parameters()):
        teacher_parameter.mul_(decay).add_(student_parameter, alpha=1.0 - decay)
        teacher_parameter.requires_grad_(False)
    teacher_buffers, student_buffers = dict(teacher.named_buffers()), dict(student.named_buffers())
    if teacher_buffers.keys() != student_buffers.keys():
        raise ValueError("teacher and student buffer structures differ")
    for name, teacher_buffer in teacher_buffers.items():
        student_buffer = student_buffers[name]
        if name == "ema_state" or name.endswith(".ema_state"):
            if not torch.is_floating_point(teacher_buffer):
                raise ValueError("ema_state buffers must be floating point")
            teacher_buffer.mul_(decay).add_(student_buffer, alpha=1.0 - decay)
        else:
            teacher_buffer.copy_(student_buffer)


def _require_finite(loss, model, *, epoch, step):
    if not torch.isfinite(loss).all():
        raise RuntimeError(f"ema non-finite loss: epoch={epoch} step={step} loss={float(loss.detach())}")
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"ema non-finite gradient: epoch={epoch} step={step} loss={float(loss.detach())} parameter={name}")


def main(argv=None):
    raw_args = build_parser().parse_args(argv)
    if raw_args.resume_checkpoint:
        raw_args.source_checkpoint = ""
    if bool(raw_args.source_checkpoint) == bool(raw_args.resume_checkpoint):
        raise ValueError("EMA requires exactly one of --source_checkpoint or --resume_checkpoint")
    checkpoint = torch.load(raw_args.resume_checkpoint or raw_args.source_checkpoint, map_location="cpu")
    expected_stage = "ema" if raw_args.resume_checkpoint else "source"
    checkpoint_preflight = (preflight_ema_resume_checkpoint(checkpoint)
                            if expected_stage == "ema"
                            else preflight_source_initialization_checkpoint(checkpoint))
    checkpoint_config = checkpoint_preflight["config"]
    args = validate_config(argparse.Namespace(**resolve_ema_config(
        raw_args, checkpoint_config, resume=expected_stage == "ema"
    )))
    _set_seed(args.model_init_seed)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    reconstruction_criteria = build_regional_reconstruction_criteria(device)
    require_coa_clip_cuda(device)
    clip_criterion, text_features = initialize_coa_clip(device)
    student = build_model_from_config(vars(args)).to(device)
    optimizer = build_coa_adam(student.parameters(), learning_rate=args.start_lr)
    if expected_stage == "source":
        load_strict_v2_state_dict(student, checkpoint_preflight["states"]["model"], label="Source model")
        teacher, start_epoch = initialize_teacher(student), 0
        source_global_step, ema_global_step, best_psnr = checkpoint_preflight["metadata"]["global_step"], 0, float("-inf")
    else:
        teacher = initialize_teacher(student)
        restored = load_ema_checkpoint(checkpoint, student, teacher, optimizer)
        start_epoch, source_global_step, ema_global_step, best_psnr = (
            restored["epoch"], restored["source_global_step"], restored["ema_global_step"], restored.get("best_psnr", float("-inf"))
        )
    geometry_generator = torch.Generator().manual_seed(args.model_init_seed + 201)
    real_hazy_dir, real_tir_dir = real_modal_dirs_from_args(args)
    real_dataset = RealMultiModalDataset(real_hazy_dir, real_tir_dir,
                                         pair_alignment_policy=args.pair_alignment_policy,
                                         tir_normalization_config=tir_normalization_config_from_args(args))
    source_dataset = SynthMultiModalDataset(args.source_anchor_data_dir, train=True, size=args.train_size,
        density_gt_semantics=args.density_gt_semantics, density_map_normalization=args.density_map_normalization,
        density_fixed_min=args.density_fixed_min, density_fixed_max=args.density_fixed_max,
        density_calibrated_min=args.density_calibrated_min, density_calibrated_max=args.density_calibrated_max,
        tir_normalization_config=tir_normalization_config_from_args(args), pair_alignment_policy=args.pair_alignment_policy,
        augmentation_seed_base=args.model_init_seed, allow_missing_density=True)
    real_loader = DataLoader(real_dataset, batch_size=args.real_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_real)
    source_loader = DataLoader(source_dataset, batch_size=args.source_anchor_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_synth)
    validation_dataset = SynthMultiModalDataset(args.validation_data_dir, train=False, size="full",
        density_gt_semantics=args.density_gt_semantics, density_map_normalization=args.density_map_normalization,
        density_fixed_min=args.density_fixed_min, density_fixed_max=args.density_fixed_max,
        density_calibrated_min=args.density_calibrated_min, density_calibrated_max=args.density_calibrated_max,
        tir_normalization_config=tir_normalization_config_from_args(args), pair_alignment_policy=args.pair_alignment_policy)
    validation_loader = DataLoader(validation_dataset, batch_size=args.validation_batch_size, shuffle=False,
                                   num_workers=args.num_workers, collate_fn=collate_synth)
    prepare_experiment_dirs(args)
    save_config(args)
    total_steps = args.epochs * args.iters_per_epoch
    real_iterator, source_iterator = cycle_batches(real_loader), cycle_batches(source_loader)
    logger = TrainingLogger(args.exp_dir or args.saved_model_dir, resume=expected_stage == "ema")
    logger.write_run_summary({
        "stage": "ema", "device": str(device), "seed": args.model_init_seed,
        "dataset_sizes": {"real": len(real_dataset), "source_anchor": len(source_dataset),
                          "validation": len(validation_dataset)},
        "batch_configuration": {"real": args.real_batch_size, "source_anchor": args.source_anchor_batch_size,
                                "validation": args.validation_batch_size, "num_workers": args.num_workers},
        "schedule": {"epochs": args.epochs, "iters_per_epoch": args.iters_per_epoch,
                     "total_steps": total_steps, "start_lr": args.start_lr, "end_lr": args.end_lr,
                     "ema_decay": args.ema_decay},
        "parameter_counts": {"student": sum(parameter.numel() for parameter in student.parameters()),
                             "teacher": sum(parameter.numel() for parameter in teacher.parameters()),
                             "rgb_encoder": sum(parameter.numel() for parameter in student.rgb_encoder.parameters()),
                             "tir_encoder": sum(parameter.numel() for parameter in student.tir_encoder.parameters())},
        "res2net_pretrained_loaded": True,
        "res2net_pretrained_path": student.rgb_encoder.pretrained_path,
        "clip_resources": {"vit_b32_loaded": True, "rn101_loaded": True, "haze_prompt_loaded": True},
        "initialization_checkpoint": raw_args.resume_checkpoint or raw_args.source_checkpoint,
        "output_directories": {"experiment": args.exp_dir, "models": args.saved_model_dir,
                               "diagnostics": args.saved_data_dir},
    })
    print(f"ema startup device={device} real={len(real_dataset)} source_anchor={len(source_dataset)} "
          f"validation={len(validation_dataset)} epochs={args.epochs} iters_per_epoch={args.iters_per_epoch} "
          f"total_steps={total_steps} student_params={sum(parameter.numel() for parameter in student.parameters()):,} "
          f"res2net={student.rgb_encoder.pretrained_path} clip=ViT-B/32,RN101,haze_prompt")
    for epoch in range(start_epoch, args.epochs):
        source_dataset.set_sampler_epoch(epoch)
        student.train(); set_batchnorm_eval(student); teacher.eval(); set_batchnorm_eval(teacher)
        for logical_step in range(args.iters_per_epoch):
            step_started = perf_counter()
            real_hazy, real_tir, _ = next(real_iterator)
            source_batch = next(source_iterator)
            schedule_step = epoch * args.iters_per_epoch + logical_step + 1
            learning_rate = set_cosine_learning_rate(
                optimizer, step=schedule_step, total_steps=total_steps, start_lr=args.start_lr,
                end_lr=args.end_lr, no_lr_sche=args.no_lr_sche,
            )
            optimizer.zero_grad(set_to_none=True)
            real = real_adaptation_loss(teacher, student, real_hazy.to(device), real_tir.to(device), geometry_generator, args,
                                        route_multiplier=1.0, clip_criterion=clip_criterion, text_features=text_features)
            source = compute_physical_mask_batch_losses(
                student, tuple(value.to(device) for value in source_batch), args, source_global_step,
                reconstruction_criteria=reconstruction_criteria,
            )
            adapt = adaptation_loss(real, source, args)
            _require_finite(adapt, student, epoch=epoch, step=ema_global_step)
            optimizer.step()
            source_global_step += 1; ema_global_step += 1
            logger.log_event(
                "train_step", epoch=epoch + 1, epoch_step=logical_step + 1,
                ema_global_step=ema_global_step, source_global_step=source_global_step,
                learning_rate=learning_rate, duration_seconds=perf_counter() - step_started,
                loss_total=adapt, loss_real=real["L_real"], loss_clip=real["L_clip"],
                loss_source_anchor=source["losses"]["total"],
                **{f"loss_real_{name.lower()}": value for name, value in real.items()},
                **{f"loss_source_{name}": value for name, value in source["losses"].items()},
                **{f"schedule_{name}": value for name, value in source["state"].items()},
            )
            if ema_global_step % 50 == 0:
                print(f"ema epoch={epoch + 1} step={ema_global_step} loss={adapt.item():.5f} "
                      f"real={real['L_real'].item():.5f} clip={real['L_clip'].item():.5f} "
                      f"anchor={source['losses']['total'].item():.5f} lr={learning_rate:.9f}")
        update_teacher_after_success(teacher, student, args.ema_decay)
        validation = evaluate_paired_validation(student, validation_loader, device, route_temperature=args.route_tau_end)
        candidate = build_ema_checkpoint(student, teacher, optimizer, epoch=epoch + 1,
            source_global_step=source_global_step, ema_global_step=ema_global_step,
            config=build_ema_checkpoint_config(checkpoint_config, args), best_psnr=max(best_psnr, validation["psnr"]))
        is_best = validation["psnr"] > best_psnr
        if args.saved_model_dir:
            output_dir = Path(args.saved_model_dir)
            best_psnr = save_best_if_improved(validation["psnr"], best_psnr, candidate, output_dir / "ema_best.pt")
            candidate["best_psnr"] = best_psnr
            torch.save(candidate, output_dir / "ema_last.pt")
        logger.log_event("validation", epoch=epoch + 1, ema_global_step=ema_global_step,
                         source_global_step=source_global_step, learning_rate=learning_rate,
                         psnr=validation["psnr"], ssim=validation["ssim"], best_psnr=best_psnr, is_best=is_best)
        print(f"ema validation epoch={epoch + 1} psnr={validation['psnr']:.4f} "
              f"ssim={validation['ssim']:.4f} best_psnr={best_psnr:.4f} is_best={is_best}")
    return student, teacher


if __name__ == "__main__":
    main()
