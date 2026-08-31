"""ResNet family vision towers (torchvision)."""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Type

import torch
import torch.nn as nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
)

from .features import FeatureMaps, parse_scale, resolve_feature_pyramids

# (builder, weights enum, channels of layer1–4 at 4x / 8x / 16x / 32x)
ResNetSpec = Tuple[Type[nn.Module], object, Tuple[int, int, int, int]]

RESNET_SPECS: Dict[str, ResNetSpec] = {
    "resnet18": (resnet18, ResNet18_Weights, (64, 128, 256, 512)),
    "resnet34": (resnet34, ResNet34_Weights, (64, 128, 256, 512)),
    "resnet50": (resnet50, ResNet50_Weights, (256, 512, 1024, 2048)),
    "resnet101": (resnet101, ResNet101_Weights, (256, 512, 1024, 2048)),
    "resnet152": (resnet152, ResNet152_Weights, (256, 512, 1024, 2048)),
}

_NATIVE_PYRAMIDS = ("4x", "8x", "16x", "32x")


class ResNetVisionTower(nn.Module):
    """ResNet through layer4; returns selected ``{'4x', '8x', '16x', '32x'}`` maps."""

    def __init__(
        self,
        name: str,
        pretrained: bool = False,
        feature_pyramids: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        builder, weights_enum, out_channels = RESNET_SPECS[name]
        weights = weights_enum.DEFAULT if pretrained else None
        backbone = builder(weights=weights)
        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.name = name
        c4, c8, c16, c32 = out_channels
        native_channels = {"4x": c4, "8x": c8, "16x": c16, "32x": c32}
        self.feature_pyramids = resolve_feature_pyramids(
            feature_pyramids, _NATIVE_PYRAMIDS, tower_name=name
        )
        self.num_channels = {key: native_channels[key] for key in self.feature_pyramids}
        self.stride = max(parse_scale(key) for key in self.feature_pyramids)

    def forward(self, x: torch.Tensor) -> FeatureMaps:
        x = self.stem(x)
        c4 = self.layer1(x)
        maps = {"4x": c4}
        need = {parse_scale(key) for key in self.feature_pyramids}
        if 8 in need or 16 in need or 32 in need:
            c8 = self.layer2(c4)
            maps["8x"] = c8
            if 16 in need or 32 in need:
                c16 = self.layer3(c8)
                maps["16x"] = c16
                if 32 in need:
                    maps["32x"] = self.layer4(c16)
        return FeatureMaps(
            {key: maps[key] for key in self.feature_pyramids},
            name=self.name,
        )
