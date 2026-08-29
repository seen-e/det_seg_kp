"""CLIP vision towers — ViT and ConvNeXt (via timm)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import timm
import torch
import torch.nn as nn

from .features import (
    FeatureMaps,
    parse_scale,
    resolve_feature_pyramids,
    scale_key,
)
from .vit_utils import extract_vit_tokens, pad_to_patch_size, tokens_to_feature_map


@dataclass(frozen=True)
class CLIPViTVisionTowerSpec:
    timm_name: str
    embed_dim: int
    patch_size: int


@dataclass(frozen=True)
class CLIPConvNeXtVisionTowerSpec:
    timm_name: str
    out_channels: int
    stride: int = 32


CLIP_VIT_SPECS: Dict[str, CLIPViTVisionTowerSpec] = {
    "clip_vit_b16_laion2b": CLIPViTVisionTowerSpec("vit_base_patch16_clip_224.laion2b", 768, 16),
    "clip_vit_b16_openai": CLIPViTVisionTowerSpec("vit_base_patch16_clip_224.openai", 768, 16),
    "clip_vit_b16_datacompxl": CLIPViTVisionTowerSpec("vit_base_patch16_clip_224.datacompxl", 768, 16),
    "clip_vit_b32_laion2b": CLIPViTVisionTowerSpec("vit_base_patch32_clip_224.laion2b", 768, 32),
    "clip_vit_l14_laion2b": CLIPViTVisionTowerSpec("vit_large_patch14_clip_224.laion2b", 1024, 14),
    "clip_vit_l14_openai": CLIPViTVisionTowerSpec("vit_large_patch14_clip_224.openai", 1024, 14),
}

CLIP_CONVNeXT_SPECS: Dict[str, CLIPConvNeXtVisionTowerSpec] = {
    "clip_convnext_base": CLIPConvNeXtVisionTowerSpec("convnext_base.clip_laion2b", 1024),
    "clip_convnext_large": CLIPConvNeXtVisionTowerSpec("convnext_large_mlp.clip_laion2b_augreg", 1536),
}


class CLIPViTVisionTower(nn.Module):
    """CLIP ViT; outputs spatial feature map from patch tokens."""

    def __init__(
        self,
        name: str,
        pretrained: bool = False,
        feature_pyramids: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        spec = CLIP_VIT_SPECS[name]
        self.name = name
        self.body = timm.create_model(
            spec.timm_name,
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=True,
        )
        native = (scale_key(spec.patch_size),)
        self.feature_pyramids = resolve_feature_pyramids(
            feature_pyramids, native, tower_name=name
        )
        self.num_channels = {key: spec.embed_dim for key in self.feature_pyramids}
        self.stride = max(parse_scale(key) for key in self.feature_pyramids)
        self._patch_size = spec.patch_size

    def forward(self, x: torch.Tensor) -> FeatureMaps:
        x, pad_h, pad_w = pad_to_patch_size(x, self._patch_size)
        tokens = extract_vit_tokens(self.body.forward_features(x), "x")
        fmap = tokens_to_feature_map(tokens, self._patch_size)
        maps = FeatureMaps(
            {scale_key(self._patch_size): fmap},
            name=self.name,
            extra={"pad_h": pad_h, "pad_w": pad_w, "patch_size": self._patch_size},
        )
        return maps.subset(self.feature_pyramids)


class CLIPConvNeXtVisionTower(nn.Module):
    """CLIP ConvNeXt; returns selected ``{'4x', '8x', '16x'}`` maps via timm features_only."""

    def __init__(
        self,
        name: str,
        pretrained: bool = False,
        feature_pyramids: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        spec = CLIP_CONVNeXT_SPECS[name]
        self.name = name
        self.body = timm.create_model(
            spec.timm_name,
            pretrained=pretrained,
            num_classes=0,
            features_only=True,
            out_indices=(0, 1, 2),
        )
        info = self.body.feature_info
        reductions = tuple(int(s) for s in info.reduction())
        channels = tuple(int(c) for c in info.channels())
        native = tuple(scale_key(s) for s in reductions)
        native_channels = {scale_key(s): c for s, c in zip(reductions, channels)}
        self.feature_pyramids = resolve_feature_pyramids(
            feature_pyramids, native, tower_name=name
        )
        self.num_channels = {key: native_channels[key] for key in self.feature_pyramids}
        self.stride = max(parse_scale(key) for key in self.feature_pyramids)
        self._native = native

    def forward(self, x: torch.Tensor) -> FeatureMaps:
        maps = list(self.body(x))
        features = FeatureMaps(
            {key: tensor for key, tensor in zip(self._native, maps)},
            name=self.name,
        )
        return features.subset(self.feature_pyramids)
