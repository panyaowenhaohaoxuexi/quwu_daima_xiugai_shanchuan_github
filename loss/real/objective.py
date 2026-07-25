"""Real-domain adaptation objective composition."""

import torch


def compute_adaptation_objective(real_loss: torch.Tensor, source_loss: torch.Tensor, *,
                                 lambda_anchor: float) -> dict[str, torch.Tensor]:
    return {"L_real": real_loss, "L_src": source_loss,
            "L_adapt": real_loss + float(lambda_anchor) * source_loss}


__all__ = ["compute_adaptation_objective"]
