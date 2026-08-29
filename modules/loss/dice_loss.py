"""Dice loss for binary / soft masks."""
from __future__ import annotations

import torch
from einops import rearrange


def dice_loss(
    inputs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    inputs = rearrange(inputs.sigmoid(), "b ... -> b (...)")
    targets = rearrange(targets, "b ... -> b (...)")
    num = 2 * (inputs * targets).sum(1)
    den = inputs.sum(1) + targets.sum(1)
    return 1 - (num + eps) / (den + eps)
