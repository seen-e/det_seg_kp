"""ResNet family vision towers (torchvision)."""
from __future__ import annotations

from typing import Dict, Tuple, Type

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

from modules.backbone.features import FeatureMaps, channels_by_scale

# (builder, weights enum, channels of layer1 / layer2 / layer3 at 4x / 8x / 16x)
ResNetSpec = Tuple[Type[nn.Module], object, Tuple[int, int, int]]

RESNET_SPECS: Dict[str, ResNetSpec] = {
    "resnet18": (resnet18, ResNet18_Weights, (64, 128, 256)),
    "resnet34": (resnet34, ResNet34_Weights, (64, 128, 256)),
    "resnet50": (resnet50, ResNet50_Weights, (256, 512, 1024)),
    "resnet101": (resnet101, ResNet101_Weights, (256, 512, 1024)),
    "resnet152": (resnet152, ResNet152_Weights, (256, 512, 1024)),
}


class ResNetVisionTower(nn.Module):
    """ResNet truncated at layer3; returns ``{'4x', '8x', '16x'}`` feature maps."""

    def __init__(self, name: str, pretrained: bool = False):
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
        self.name = name
        c4, c8, c16 = out_channels
        self.num_channels = channels_by_scale({4: c4, 8: c8, 16: c16})
        self.stride = 16

    def forward(self, x: torch.Tensor) -> FeatureMaps:
        x = self.stem(x)
        c4 = self.layer1(x)
        c8 = self.layer2(c4)
        c16 = self.layer3(c8)
        return FeatureMaps(
            {
                "4x": c4,
                "8x": c8,
                "16x": c16,
            },
            name=self.name,
        )
