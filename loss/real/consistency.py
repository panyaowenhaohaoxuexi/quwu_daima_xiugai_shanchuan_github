"""Detached teacher-stability weights and real-domain consistency losses."""

import torch

from loss.common.masked import weighted_bce, weighted_l1


def stability_weights(j_a: torch.Tensor, j_b: torch.Tensor, m_a: torch.Tensor, m_b: torch.Tensor,
                      r_a: torch.Tensor, r_b: torch.Tensor, sigma_j: float, sigma_m: float,
                      sigma_r: float, minimum: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if min(sigma_j, sigma_m, sigma_r) <= 0:
        raise ValueError("EMA sigmas must be positive")
    w_j = torch.exp(-(j_a - j_b).abs().mean(dim=1, keepdim=True) / sigma_j)
    w_m = torch.exp(-(m_a - m_b).abs() / sigma_m)
    w_r = torch.exp(-(r_a - r_b).abs() / sigma_r)
    return tuple(weight.clamp(minimum, 1.0).detach() for weight in (w_j, w_m, w_r))


def real_consistency_loss(j_student: torch.Tensor, j_target: torch.Tensor, m_student: torch.Tensor,
                          m_target: torch.Tensor, r_student: torch.Tensor, r_target: torch.Tensor,
                          w_j: torch.Tensor, w_m: torch.Tensor, w_r: torch.Tensor,
                          lambda_j: float = 1.0, lambda_m: float = 1.0,
                          lambda_r: float = 1.0) -> dict[str, torch.Tensor]:
    l_j = weighted_l1(j_student, j_target, w_j)
    l_m = weighted_l1(m_student, m_target, w_m)
    l_r = weighted_bce(r_student, r_target, w_r)
    return {"L_J": l_j, "L_M": l_m, "L_R": l_r,
            "L_real": lambda_j * l_j + lambda_m * l_m + lambda_r * l_r}


__all__ = ["stability_weights", "weighted_l1", "weighted_bce", "real_consistency_loss"]
