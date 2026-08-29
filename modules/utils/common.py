"""Shared helpers: seeding, box residual decode, Gaussian heatmaps."""
import random
from typing import Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch (including all CUDA devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Numerically stable inverse of sigmoid for values in (0, 1)."""
    x = x.clamp(0.0, 1.0)
    return torch.log(x.clamp(min=eps) / (1.0 - x).clamp(min=eps))


def decode_boxes(delta: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Residual box decode: sigmoid(delta + inverse_sigmoid(reference)).

    ``reference`` is (..., 2) centers or (..., 4) cxcywh; ``delta`` is always 4D.
    """
    inv = inverse_sigmoid(reference)
    if reference.shape[-1] == 4:
        return (delta + inv).sigmoid()
    out = delta.clone()
    out[..., :2] = delta[..., :2] + inv
    return out.sigmoid()


def gaussian_heatmap(
    center: Tuple[float, float],
    size: Tuple[int, int],
    sigma: float = 2.0,
) -> np.ndarray:
    """Generate a 2D Gaussian heatmap centered at (x, y)."""
    h, w = size
    x, y = center
    xs = np.arange(0, w, dtype=np.float32)
    ys = np.arange(0, h, dtype=np.float32)[:, None]
    heatmap = np.exp(-((xs - x) ** 2 + (ys - y) ** 2) / (2 * sigma ** 2))
    return heatmap
