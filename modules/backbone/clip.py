"""CLIP vision towers — ViT and ConvNeXt (via timm)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import timm
import torch
import torch.nn as nn

from modules.backbone.features import FeatureMaps, scale_key
from modules.backbone.vit_utils import extract_vit_tokens, pad_to_patch_size, tokens_to_feature_map


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

    def __init__(self, name: str, pretrained: bool = False):
        super().__init__()
        spec = CLIP_VIT_SPECS[name]
        self.name = name
        self.body = timm.create_model(
            spec.timm_name,
            pretrained=pretrained,
            num_classes=0,
            dynamic_img_size=True,
        )
        self.stride = spec.patch_size
        self.num_channels = {scale_key(spec.patch_size): spec.embed_dim}
        self._patch_size = spec.patch_size

    def forward(self, x: torch.Tensor) -> FeatureMaps:
        x, pad_h, pad_w = pad_to_patch_size(x, self._patch_size)
        tokens = extract_vit_tokens(self.body.forward_features(x), "x")
        fmap = tokens_to_feature_map(tokens, self._patch_size)
        return FeatureMaps(
            {scale_key(self._patch_size): fmap},
            name=self.name,
            extra={"pad_h": pad_h, "pad_w": pad_w, "patch_size": self._patch_size},
        )


class CLIPConvNeXtVisionTower(nn.Module):
    """CLIP ConvNeXt; returns ``{'4x', '8x', '16x'}`` maps via timm features_only."""

    def __init__(self, name: str, pretrained: bool = False):
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
        self.num_channels = {scale_key(s): c for s, c in zip(reductions, channels)}
        self.stride = int(reductions[-1])

    def forward(self, x: torch.Tensor) -> FeatureMaps:
        maps = list(self.body(x))
        features = {
            key: tensor
            for key, tensor in zip(self.num_channels.keys(), maps)
        }
        return FeatureMaps(features, name=self.name)
