"""DINOv2 ViT vision towers (via timm)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import timm
import torch
import torch.nn as nn

from modules.backbone.features import (
    FeatureMaps,
    parse_scale,
    resolve_feature_pyramids,
    scale_key,
)
from modules.backbone.vit_utils import extract_vit_tokens, pad_to_patch_size, tokens_to_feature_map


@dataclass(frozen=True)
class DINOV2VisionTowerSpec:
    timm_name: str
    embed_dim: int
    patch_size: int


DINOV2_SPECS: Dict[str, DINOV2VisionTowerSpec] = {
    "dinov2_vits14": DINOV2VisionTowerSpec("vit_small_patch14_dinov2.lvd142m", 384, 14),
    "dinov2_vitb14": DINOV2VisionTowerSpec("vit_base_patch14_dinov2.lvd142m", 768, 14),
    "dinov2_vitl14": DINOV2VisionTowerSpec("vit_large_patch14_dinov2.lvd142m", 1024, 14),
    "dinov2_vitg14": DINOV2VisionTowerSpec("vit_giant_patch14_dinov2.lvd142m", 1536, 14),
    "dinov2_vits14_reg": DINOV2VisionTowerSpec("vit_small_patch14_reg4_dinov2.lvd142m", 384, 14),
    "dinov2_vitb14_reg": DINOV2VisionTowerSpec("vit_base_patch14_reg4_dinov2.lvd142m", 768, 14),
    "dinov2_vitl14_reg": DINOV2VisionTowerSpec("vit_large_patch14_reg4_dinov2.lvd142m", 1024, 14),
    "dinov2_vitg14_reg": DINOV2VisionTowerSpec("vit_giant_patch14_reg4_dinov2.lvd142m", 1536, 14),
}


class DINOV2VisionTower(nn.Module):
    """DINOv2 ViT; outputs spatial feature map from patch tokens."""

    def __init__(
        self,
        name: str,
        pretrained: bool = False,
        feature_pyramids: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        spec = DINOV2_SPECS[name]
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
        tokens = extract_vit_tokens(self.body.forward_features(x), "x_norm_patchtokens")
        fmap = tokens_to_feature_map(tokens, self._patch_size)
        maps = FeatureMaps(
            {scale_key(self._patch_size): fmap},
            name=self.name,
            extra={"pad_h": pad_h, "pad_w": pad_w, "patch_size": self._patch_size},
        )
        return maps.subset(self.feature_pyramids)
