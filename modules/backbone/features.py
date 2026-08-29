"""Multi-scale feature maps returned by vision towers."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch


def scale_key(stride: int) -> str:
    """Map a downsample factor to a dict key, e.g. ``4`` -> ``'4x'``."""
    return f"{int(stride)}x"


def parse_scale(key: str) -> int:
    """Parse ``'4x'`` / ``'4'`` into the integer downsample factor."""
    text = str(key).strip().lower()
    if text.endswith("x"):
        text = text[:-1]
    return int(text)


def channels_by_scale(stride_to_channels: Mapping[int, int]) -> Dict[str, int]:
    return {scale_key(stride): int(ch) for stride, ch in stride_to_channels.items()}


class FeatureMaps(dict):
    """Dict of spatial feature maps keyed by downsample factor.

    Typical keys are ``'2x'``, ``'4x'``, ``'8x'``, ``'16x'``. Extra metadata
    lives on ``name`` / ``extra`` so it does not collide with tensor keys::

        feats = tower(images)
        c4 = feats["4x"]          # (B, C, H/4, W/4)
        for key in feats.sorted_keys():
            ...
        feats.channels["8x"]
        feats.extra["pad_h"]
    """

    def __init__(
        self,
        features: Optional[Mapping[str, torch.Tensor]] = None,
        *,
        name: str = "",
        extra: Optional[Mapping[str, Any]] = None,
        **tensors: torch.Tensor,
    ):
        merged: Dict[str, torch.Tensor] = {}
        if features:
            merged.update(features)
        merged.update(tensors)
        super().__init__(merged)
        self.name = name
        self.extra: Dict[str, Any] = dict(extra) if extra else {}

    def sorted_keys(self) -> List[str]:
        """Keys from finest (smallest stride) to coarsest."""
        return sorted(self.keys(), key=parse_scale)

    @property
    def strides(self) -> Dict[str, int]:
        return {key: parse_scale(key) for key in self}

    @property
    def channels(self) -> Dict[str, int]:
        return {key: int(value.shape[1]) for key, value in self.items()}

    @property
    def finest_key(self) -> str:
        return self.sorted_keys()[0]

    @property
    def coarsest_key(self) -> str:
        return self.sorted_keys()[-1]

    def finest(self) -> torch.Tensor:
        return self[self.finest_key]

    def coarsest(self) -> torch.Tensor:
        return self[self.coarsest_key]

    def tensors(self) -> "FeatureMaps":
        """Drop any non-tensor entries if a plain dict mixed in metadata."""
        return FeatureMaps(
            {k: v for k, v in self.items() if torch.is_tensor(v)},
            name=self.name,
            extra=self.extra,
        )

    def __repr__(self) -> str:
        parts = []
        for key in self.sorted_keys():
            value = self[key]
            if torch.is_tensor(value):
                parts.append(f"{key}: {tuple(value.shape)}")
            else:
                parts.append(f"{key}: {type(value).__name__}")
        body = ", ".join(parts) if parts else "empty"
        suffix = f", extra={self.extra}" if self.extra else ""
        name = f"{self.name} " if self.name else ""
        return f"FeatureMaps({name}{body}{suffix})"


def as_feature_maps(
    features: Mapping[str, Any] | FeatureMaps | torch.Tensor,
    *,
    name: str = "",
    default_stride: Optional[int] = None,
) -> FeatureMaps:
    """Normalize tower output to ``FeatureMaps``.

    A lone tensor is stored under ``'{default_stride}x'``. Non-tensor dict
    values are moved into ``extra``.
    """
    if isinstance(features, FeatureMaps):
        return features
    if torch.is_tensor(features):
        if default_stride is None:
            raise TypeError("a single feature map needs default_stride to build a key")
        return FeatureMaps({scale_key(default_stride): features}, name=name)

    tensors: Dict[str, torch.Tensor] = {}
    extra: Dict[str, Any] = {}
    for key, value in features.items():
        if torch.is_tensor(value):
            tensors[str(key)] = value
        else:
            extra[str(key)] = value
    return FeatureMaps(tensors, name=name, extra=extra)


def iter_scales(features: Mapping[str, torch.Tensor]) -> Iterable[tuple[str, int, torch.Tensor]]:
    """Yield ``(key, stride, tensor)`` from finest to coarsest."""
    maps = as_feature_maps(features)
    for key in maps.sorted_keys():
        yield key, parse_scale(key), maps[key]
