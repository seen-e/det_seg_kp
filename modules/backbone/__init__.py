"""Vision tower factory."""
from __future__ import annotations

from typing import Any, Mapping, Union

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
from modules.config import VisionTowerConfig

_ALL_SPECS = {**RESNET_SPECS, **DINOV2_SPECS, **CLIP_VIT_SPECS, **CLIP_CONVNeXT_SPECS}

# Family aliases used by ``VisionTowerConfig.type``.
_FAMILY_DEFAULTS = {
    "resnet": "resnet50",
    "dinov2": "dinov2_vitl14",
    "clip_vit": "clip_vit_l14_openai",
    "clip_convnext": "clip_convnext_base",
}

VisionTowerSpec = Union[VisionTowerConfig, Mapping[str, Any], str]


def list_vision_towers() -> list[str]:
    return sorted(_ALL_SPECS)


def parse_vision_tower_cfg(cfg: VisionTowerSpec) -> VisionTowerConfig:
    """Normalize dict / str / dataclass into :class:`VisionTowerConfig`."""
    if isinstance(cfg, VisionTowerConfig):
        return cfg
    if isinstance(cfg, str):
        return VisionTowerConfig(type=cfg, feature_pyramids=None, pretrained=True)
    if isinstance(cfg, Mapping):
        if "type" not in cfg:
            raise ValueError("vision_tower dict needs 'type'")
        kwargs: dict[str, Any] = {"type": cfg["type"]}
        if "feature_pyramids" in cfg:
            pyramids = cfg["feature_pyramids"]
            kwargs["feature_pyramids"] = None if pyramids is None else list(pyramids)
        if "pretrained" in cfg:
            kwargs["pretrained"] = bool(cfg["pretrained"])
        return VisionTowerConfig(**kwargs)
    raise TypeError(
        f"vision_tower must be VisionTowerConfig, dict, or str; got {type(cfg).__name__}"
    )


def resolve_vision_tower_type(type_name: str) -> str:
    """Map ``type`` to a registry key (``resnet`` → ``resnet50``)."""
    key = type_name.lower().strip()
    if key in _ALL_SPECS:
        return key
    if key in _FAMILY_DEFAULTS:
        return _FAMILY_DEFAULTS[key]
    raise ValueError(
        f"Unknown vision tower type '{type_name}'. "
        f"Supported names: {', '.join(list_vision_towers())}. "
        f"Family aliases: {', '.join(sorted(_FAMILY_DEFAULTS))}"
    )


def build_vision_tower(cfg: VisionTowerSpec, pretrained: bool | None = None) -> nn.Module:
    """
    Build a vision tower from a config dict / :class:`VisionTowerConfig`.

    Each vision tower exposes ``num_channels`` as ``{'4x': C, '8x': C, ...}``
    (or ``{'14x': C}`` for ViT-L) covering only ``feature_pyramids``, and
    returns a :class:`FeatureMaps` dict keyed the same way. ``stride`` is the
    coarsest selected downsample factor.

    ``pretrained`` overrides ``cfg.pretrained`` when not None (legacy kwarg).
    """
    tower_cfg = parse_vision_tower_cfg(cfg)
    key = resolve_vision_tower_type(tower_cfg.type)
    use_pretrained = tower_cfg.pretrained if pretrained is None else pretrained
    pyramids = tower_cfg.feature_pyramids
    if key in RESNET_SPECS:
        return ResNetVisionTower(key, use_pretrained, pyramids)
    if key in DINOV2_SPECS:
        return DINOV2VisionTower(key, use_pretrained, pyramids)
    if key in CLIP_VIT_SPECS:
        return CLIPViTVisionTower(key, use_pretrained, pyramids)
    if key in CLIP_CONVNeXT_SPECS:
        return CLIPConvNeXtVisionTower(key, use_pretrained, pyramids)
    raise ValueError(
        f"Unknown vision tower '{tower_cfg.type}'. Supported: {', '.join(list_vision_towers())}"
    )


__all__ = [
    "FeatureMaps",
    "as_feature_maps",
    "build_vision_tower",
    "list_vision_towers",
    "parse_scale",
    "parse_vision_tower_cfg",
    "resolve_vision_tower_type",
    "scale_key",
]
