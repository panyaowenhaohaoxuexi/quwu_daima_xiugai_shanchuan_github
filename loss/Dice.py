import torch.nn as nn
import torch
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1
        self.balance = 1.1

    def forward(self, inputs, targets):
        n, c, h, w = inputs.size()

        input_flat = inputs.view(-1)
        target_flat = targets.view(-1)
        intersecion = input_flat * target_flat
        unionsection = input_flat.pow(2).sum() + target_flat.pow(2).sum() + self.smooth
        loss = unionsection/(2 * intersecion.sum() + self.smooth)
        loss = loss.sum()

        return loss