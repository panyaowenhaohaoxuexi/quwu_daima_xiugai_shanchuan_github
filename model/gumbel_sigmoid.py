import torch
import torch.nn as nn


class GumbelSigmoidBinarizer(nn.Module):
    """Gumbel-Sigmoid binarizer where 1 means IR completion region."""

    def __init__(self, tau=1.0, hard=True, threshold=0.5, eps=1e-6):
        super().__init__()
        self.tau = float(tau)
        self.hard = hard
        self.threshold = threshold
        self.eps = eps

    def set_tau(self, tau):
        self.tau = float(tau)

    def _prob_to_logits(self, x):
        x = x.clamp(self.eps, 1.0 - self.eps)
        return torch.log(x) - torch.log1p(-x)

    def forward(self, x, is_logits=True):
        if is_logits:
            logits = x
            mask_prob = torch.sigmoid(logits)
        else:
            mask_prob = x.clamp(self.eps, 1.0 - self.eps)
            logits = self._prob_to_logits(mask_prob)

        if self.training:
            u = torch.rand_like(logits).clamp(self.eps, 1.0 - self.eps)
            g = torch.log(u) - torch.log1p(-u)
            soft = torch.sigmoid((logits + g) / max(self.tau, self.eps))
            if self.hard:
                hard = (soft >= self.threshold).float()
                binary = hard.detach() - soft.detach() + soft
            else:
                binary = soft
        else:
            binary = (mask_prob >= self.threshold).float()

        return mask_prob, binary
