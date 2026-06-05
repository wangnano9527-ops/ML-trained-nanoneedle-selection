from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    intersection = torch.sum(probs * targets, dim=dims)
    denominator = torch.sum(probs, dim=dims) + torch.sum(targets, dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, *, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = float(bce_weight)
        self.dice_weight = float(dice_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        dice = dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice


@torch.no_grad()
def segmentation_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> dict[str, float]:
    preds = (torch.sigmoid(logits) >= threshold).float()
    targets = targets.float()
    tp = torch.sum(preds * targets)
    fp = torch.sum(preds * (1.0 - targets))
    fn = torch.sum((1.0 - preds) * targets)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    return {
        "precision": float(precision.detach().cpu()),
        "recall": float(recall.detach().cpu()),
        "dice": float(dice.detach().cpu()),
    }

