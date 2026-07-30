"""Geometry-equivariant real-domain consistency losses shared by EMA and UDA."""

from dataclasses import dataclass

import torch

from loss.ema import real_consistency_loss, stability_weights


@dataclass(frozen=True)
class Geometry:
    rot90_k: int
    horizontal_flip: bool

    def apply(self, tensor):
        tensor = torch.rot90(tensor, self.rot90_k, dims=(-2, -1))
        return torch.flip(tensor, dims=(-1,)) if self.horizontal_flip else tensor

    def inverse(self, tensor):
        tensor = torch.flip(tensor, dims=(-1,)) if self.horizontal_flip else tensor
        return torch.rot90(tensor, (-self.rot90_k) % 4, dims=(-2, -1))


def sample_geometry(generator):
    return Geometry(int(torch.randint(0, 4, (), generator=generator)),
                    bool(torch.randint(0, 2, (), generator=generator)))


def real_adaptation_loss(teacher, student, hazy, tir, generator, args, *,
                         route_multiplier=1.0, clip_criterion=None, text_features=None):
    """Compute real-domain consistency while allowing route consistency warm-up."""
    if not 0.0 <= float(route_multiplier) <= 1.0:
        raise ValueError("route_multiplier must be in [0, 1]")
    transforms = tuple(sample_geometry(generator) for _ in range(3))
    with torch.no_grad():
        a = teacher(transforms[0].apply(hazy), transforms[0].apply(tir),
                    route_temperature=args.route_tau_end, route_mode="hard")
        b = teacher(transforms[1].apply(hazy), transforms[1].apply(tir),
                    route_temperature=args.route_tau_end, route_mode="hard")
    s = student(transforms[2].apply(hazy), transforms[2].apply(tir),
                route_temperature=args.route_tau_end, route_mode="hard")
    j_a, j_b = transforms[0].inverse(a["pred_clear"]), transforms[1].inverse(b["pred_clear"])
    m_a, m_b = transforms[0].inverse(a["density_map"]), transforms[1].inverse(b["density_map"])
    r_a, r_b = transforms[0].inverse(a["route_soft"]), transforms[1].inverse(b["route_soft"])
    weights = stability_weights(j_a, j_b, m_a, m_b, r_a, r_b, args.ema_sigma_j, args.ema_sigma_m,
                                args.ema_sigma_r, args.ema_stability_min_weight)
    student_clear = transforms[2].inverse(s["pred_clear"])
    losses = real_consistency_loss(
        student_clear, 0.5 * (j_a + j_b), transforms[2].inverse(s["density_map"]),
        0.5 * (m_a + m_b), transforms[2].inverse(s["route_soft"]), 0.5 * (r_a + r_b), *weights,
        lambda_j=args.lambda_ema_j, lambda_m=args.lambda_ema_m,
        lambda_r=args.lambda_ema_r * float(route_multiplier),
    )
    losses["L_R"] = losses["L_R"] * float(route_multiplier)
    losses["L_clip"] = (clip_criterion(student_clear, text_features)
                        if clip_criterion is not None else student_clear.new_zeros(()))
    return losses
