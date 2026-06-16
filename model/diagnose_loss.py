import torch
import torch.nn.functional as F


def density_loss(C_pred, density_gt):
    # density_gt: (B,1,H,W) in [0,1], high means dense haze.
    return F.l1_loss(C_pred, density_gt)


def mask_loss(mask_logits, mask_gt, dice_weight=0.5):
    # mask_gt: (B,1,H,W) in {0,1}.
    bce = F.binary_cross_entropy_with_logits(mask_logits, mask_gt)
    prob = torch.sigmoid(mask_logits)
    inter = (prob * mask_gt).sum(dim=[1, 2, 3])
    dice = 1 - (2 * inter + 1e-6) / (
        prob.sum(dim=[1, 2, 3]) + mask_gt.sum(dim=[1, 2, 3]) + 1e-6
    )
    return bce + dice_weight * dice.mean()
