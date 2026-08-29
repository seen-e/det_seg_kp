"""CornerNet / CenterNet heatmap focal loss (sparse Gaussian peaks)."""
from __future__ import annotations

import torch


def centernet_heatmap_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
    peak_thresh: float = 0.8,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Distinct from RetinaNet ``sigmoid_focal_loss``.

    - positives: near-peak pixels (Y >= ``peak_thresh``)
    - non-peaks: ``-(1-Y)^β * p^α * log(1-p)``
    - normalize by #positive pixels, not full-map mean

    Always computed in float32 (bf16 autocast makes ``log``/``pow`` overflow to Inf).

    ``inputs``: logits; ``targets``: Gaussian heatmap in [0, 1].
    """
    # Autocast-safe: bf16 log/pow on heatmap easily overflows to Inf.
    pred = inputs.float().clamp(-10.0, 10.0).sigmoid().clamp(min=eps, max=1.0 - eps)
    targets = targets.float()

    pos_mask = targets >= peak_thresh
    neg_mask = ~pos_mask
    neg_weights = (1.0 - targets).clamp(min=0.0).pow(beta)

    pos_loss = -(1.0 - pred).pow(alpha) * pred.log() * pos_mask.to(pred.dtype)
    neg_loss = (
        -pred.pow(alpha) * (1.0 - pred).log() * neg_weights * neg_mask.to(pred.dtype)
    )

    num_pos = pos_mask.to(pred.dtype).sum().clamp(min=1.0)
    loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
    # Guard only; fp32 path above should already be finite under bf16 autocast.
    return torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0)
