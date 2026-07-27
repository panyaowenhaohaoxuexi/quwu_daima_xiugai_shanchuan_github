"""Real-domain EMA adaptation entry point for fog-routed RGB--TIR dehazing."""

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data.data_loader import RealMultiModalDataset, SynthMultiModalDataset, collate_real, collate_synth
from loss.ema import real_consistency_loss, stability_weights
from option.EMA import (build_ema_checkpoint_config, build_parser, prepare_experiment_dirs,
                        resolve_ema_config, save_config, tir_normalization_config_from_args, validate_config)
from training.source import OmegaSampler, compute_source_batch_losses
from utils.checkpoint import (build_ema_checkpoint, build_model_from_config, load_ema_checkpoint,
                              require_stage)


def set_batchnorm_eval(module):
    for child in module.modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            child.eval()


@dataclass(frozen=True)
class _Geometry:
    rot90_k: int
    horizontal_flip: bool

    def apply(self, tensor):
        tensor = torch.rot90(tensor, self.rot90_k, dims=(-2, -1))
        return torch.flip(tensor, dims=(-1,)) if self.horizontal_flip else tensor

    def inverse(self, tensor):
        tensor = torch.flip(tensor, dims=(-1,)) if self.horizontal_flip else tensor
        return torch.rot90(tensor, (-self.rot90_k) % 4, dims=(-2, -1))


def _sample_geometry(generator):
    return _Geometry(int(torch.randint(0, 4, (), generator=generator)),
                     bool(torch.randint(0, 2, (), generator=generator)))


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


def _real_loss(teacher, student, hazy, tir, generator, args):
    transforms = tuple(_sample_geometry(generator) for _ in range(3))
    with torch.no_grad():
        a = teacher(transforms[0].apply(hazy), transforms[0].apply(tir), route_temperature=args.route_tau_end, route_mode="hard")
        b = teacher(transforms[1].apply(hazy), transforms[1].apply(tir), route_temperature=args.route_tau_end, route_mode="hard")
    s = student(transforms[2].apply(hazy), transforms[2].apply(tir), route_temperature=args.route_tau_end, route_mode="hard")
    j_a, j_b = transforms[0].inverse(a["pred_clear"]), transforms[1].inverse(b["pred_clear"])
    m_a, m_b = transforms[0].inverse(a["density_map"]), transforms[1].inverse(b["density_map"])
    r_a, r_b = transforms[0].inverse(a["route_soft"]), transforms[1].inverse(b["route_soft"])
    weights = stability_weights(j_a, j_b, m_a, m_b, r_a, r_b, args.ema_sigma_j, args.ema_sigma_m,
                               args.ema_sigma_r, args.ema_stability_min_weight)
    return real_consistency_loss(transforms[2].inverse(s["pred_clear"]), 0.5 * (j_a + j_b),
                                 transforms[2].inverse(s["density_map"]), 0.5 * (m_a + m_b),
                                 transforms[2].inverse(s["route_soft"]), 0.5 * (r_a + r_b), *weights,
                                 lambda_j=args.lambda_ema_j, lambda_m=args.lambda_ema_m, lambda_r=args.lambda_ema_r)


def main(argv=None):
    raw_args = build_parser().parse_args(argv)
    if bool(raw_args.source_checkpoint) == bool(raw_args.resume_checkpoint):
        raise ValueError("EMA requires exactly one of --source_checkpoint or --resume_checkpoint")
    checkpoint = torch.load(raw_args.resume_checkpoint or raw_args.source_checkpoint, map_location="cpu")
    expected_stage = "ema" if raw_args.resume_checkpoint else "source"
    require_stage(checkpoint, expected_stage)
    args = validate_config(argparse.Namespace(**resolve_ema_config(
        raw_args, checkpoint["config"], resume=expected_stage == "ema"
    )))
    prepare_experiment_dirs(args)
    save_config(args)
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    student = build_model_from_config(vars(args)).to(device)
    optimizer = AdamW(student.parameters(), lr=args.learning_rate)
    if expected_stage == "source":
        student.load_state_dict(checkpoint["model"], strict=True)
        teacher, start_epoch = initialize_teacher(student), 0
        source_global_step, ema_global_step = int(checkpoint["global_step"]), 0
    else:
        teacher = initialize_teacher(student)
        restored = load_ema_checkpoint(checkpoint, student, teacher, optimizer)
        start_epoch, source_global_step, ema_global_step = restored["epoch"], restored["source_global_step"], restored["ema_global_step"]
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.learning_rate
    real_dataset = RealMultiModalDataset(f"{args.real_data_dir}/hazy", f"{args.real_data_dir}/tir",
                                         pair_alignment_policy=args.pair_alignment_policy,
                                         tir_normalization_config=tir_normalization_config_from_args(args))
    source_dataset = SynthMultiModalDataset(args.source_anchor_data_dir, train=True, size=args.train_size,
        density_gt_semantics=args.density_gt_semantics, density_map_normalization=args.density_map_normalization,
        density_fixed_min=args.density_fixed_min, density_fixed_max=args.density_fixed_max,
        density_calibrated_min=args.density_calibrated_min, density_calibrated_max=args.density_calibrated_max,
        tir_normalization_config=tir_normalization_config_from_args(args), pair_alignment_policy=args.pair_alignment_policy,
        augmentation_seed_base=args.model_init_seed)
    real_loader = DataLoader(real_dataset, batch_size=args.real_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_real)
    source_loader = DataLoader(source_dataset, batch_size=args.source_anchor_batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_synth)
    geometry_generator = torch.Generator(device=device).manual_seed(args.model_init_seed + 201)
    omega_generator = torch.Generator(device=device).manual_seed(args.model_init_seed + 101)
    omega_sampler = OmegaSampler(args.omega_regions_per_image, args.omega_min_area, args.omega_max_area, args.model_init_seed)
    empty_omega_streak = 0
    for epoch in range(start_epoch, args.epochs):
        source_dataset.set_sampler_epoch(epoch)
        student.train(); set_batchnorm_eval(student); teacher.eval(); set_batchnorm_eval(teacher)
        source_iterator = iter(source_loader)
        for real_hazy, real_tir, _ in real_loader:
            try:
                source_batch = next(source_iterator)
            except StopIteration:
                source_iterator = iter(source_loader)
                source_batch = next(source_iterator)
            optimizer.zero_grad(set_to_none=True)
            real = _real_loss(teacher, student, real_hazy.to(device), real_tir.to(device), geometry_generator, args)
            source = compute_source_batch_losses(student, tuple(value.to(device) for value in source_batch), args,
                                                 omega_sampler, source_global_step, force_anchor_mode=True,
                                                 omega_generator=omega_generator)
            adapt = real["L_real"] + args.lambda_anchor * source["losses"]["total"]
            _require_finite(adapt, student, epoch=epoch, step=ema_global_step)
            optimizer.step()
            update_teacher_after_success(teacher, student, args.ema_decay)
            route_supervision_enabled = args.lambda_router * source["state"]["lambda_route"] > 0
            empty_omega_streak = empty_omega_streak + 1 if (
                route_supervision_enabled and not source["route_supervision"]["valid_q_region_count"]
            ) else 0
            if empty_omega_streak >= args.max_consecutive_empty_omega_steps:
                raise RuntimeError(f"ema empty Omega limit: epoch={epoch} step={ema_global_step}")
            source_global_step += 1; ema_global_step += 1
        if args.saved_model_dir:
            torch.save(build_ema_checkpoint(student, teacher, optimizer, epoch=epoch + 1,
                source_global_step=source_global_step, ema_global_step=ema_global_step,
                config=build_ema_checkpoint_config(checkpoint["config"], args)), Path(args.saved_model_dir) / "ema_last.pt")
    return student, teacher


if __name__ == "__main__":
    main()
