"""Vision tower factory."""
from __future__ import annotations

import torch.nn as nn

from modules.backbone.clip import (
    CLIP_CONVNeXT_SPECS,
    CLIP_VIT_SPECS,
    CLIPConvNeXtVisionTower,
    CLIPViTVisionTower,
)
from modules.backbone.dinov2 import DINOV2_SPECS, DINOV2VisionTower
from modules.backbone.features import FeatureMaps, as_feature_maps, parse_scale, scale_key
from modules.backbone.resnet import RESNET_SPECS, ResNetVisionTower

_ALL_SPECS = {**RESNET_SPECS, **DINOV2_SPECS, **CLIP_VIT_SPECS, **CLIP_CONVNeXT_SPECS}


def list_vision_towers() -> list[str]:
    return sorted(_ALL_SPECS)


def build_vision_tower(name: str, pretrained: bool = False) -> nn.Module:
    """
    Build a vision tower by name.

    Each vision tower exposes ``num_channels`` as ``{'4x': C, '8x': C, ...}``
    and returns a :class:`FeatureMaps` dict of spatial tensors keyed the same
    way. ``stride`` is the coarsest downsample factor.
    """
    key = name.lower()
    if key in RESNET_SPECS:
        return ResNetVisionTower(key, pretrained)
    if key in DINOV2_SPECS:
        return DINOV2VisionTower(key, pretrained)
    if key in CLIP_VIT_SPECS:
        return CLIPViTVisionTower(key, pretrained)
    if key in CLIP_CONVNeXT_SPECS:
        return CLIPConvNeXtVisionTower(key, pretrained)
    raise ValueError(
        f"Unknown vision tower '{name}'. Supported: {', '.join(list_vision_towers())}"
    )


__all__ = [
    "FeatureMaps",
    "as_feature_maps",
    "build_vision_tower",
    "list_vision_towers",
    "parse_scale",
    "scale_key",
]
