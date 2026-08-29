"""RetinaNet-style sigmoid focal loss (soft targets OK)."""
from __future__ import annotations

from typing import Optional

import torch


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    pos_thresh: Optional[float] = None,
) -> torch.Tensor:
    """RetinaNet-style sigmoid focal loss.

    Used for masks (``pos_thresh=None``, mean over pixels) and KP heatmaps
    (``pos_thresh`` set → sum / #{Y >= thresh}). Distinct from
    :func:`centernet_heatmap_loss`.
    """
    prob = inputs.sigmoid()
    ce = torch.nn.functional.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if pos_thresh is None:
        return loss.mean()
    num_pos = (targets >= pos_thresh).to(loss.dtype).sum().clamp(min=1.0)
    return loss.sum() / num_pos
