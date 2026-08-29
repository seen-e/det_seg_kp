"""Shared helpers for ViT-style vision towers."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from einops import rearrange


def pad_to_patch_size(x: torch.Tensor, patch_size: int) -> tuple[torch.Tensor, int, int]:
    """Pad bottom/right so H and W are divisible by patch_size."""
    _, _, h, w = x.shape
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h))
    return x, pad_h, pad_w


def tokens_to_feature_map(
    tokens: torch.Tensor,
    patch_size: int,
    *,
    has_cls_token: bool = True,
) -> torch.Tensor:
    """Reshape (B, N, C) tokens to (B, C, H, W).

    If ``has_cls_token``, drop ``tokens[:, 0]``. Pass ``False`` for patch-only
    tensors (e.g. DINOv2 ``x_norm_patchtokens``).
    """
    if has_cls_token:
        tokens = tokens[:, 1:]
    _, n, _ = tokens.shape
    side = int(n**0.5)
    if side * side != n:
        raise ValueError(f"Token count {n} is not a perfect square after removing CLS.")
    return rearrange(tokens, "b (h w) c -> b c h w", h=side, w=side)


def extract_vit_tokens(features: torch.Tensor | dict, token_key: str) -> torch.Tensor:
    """Pull token tensor from a timm feature dict, with patchtoken / ``x`` fallbacks."""
    if not isinstance(features, dict):
        return features
    if token_key in features:
        return features[token_key]
    for fallback in ("x_norm_patchtokens", "x"):
        if fallback in features:
            return features[fallback]
    return next(iter(features.values()))
