"""FPN pixel decoder (MaskFormer / Mask2Former style)."""
from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbone import FeatureMaps, as_feature_maps, parse_scale
from ..config import PixelDecoderConfig


def _gn_groups(num_channels: int, max_groups: int = 32) -> int:
    for groups in (max_groups, 16, 8, 4, 2, 1):
        if num_channels % groups == 0:
            return groups
    return 1


class PixelDecoder(nn.Module):
    """FPN pixel decoder (MaskFormer BasePixelDecoder / Mask2Former style).

    Consumes a :class:`FeatureMaps` dict (e.g. ``{'4x', '8x', '16x'}``).
    Top-down laterals fuse them; the finest map is optionally upsampled to
    ``out_stride`` (default 4) and projected to ``pixel_dim`` for mask / kp heads.

    Returns:
      pixel_features: (B, pixel_dim, H/out_stride, W/out_stride)
      fpn: FeatureMaps of fused maps at each backbone scale
    """

    def __init__(self, cfg: PixelDecoderConfig):
        super().__init__()
        in_channels = dict(cfg.in_channels)
        if not in_channels:
            raise ValueError("PixelDecoder needs at least one backbone scale in in_channels")

        pixel_dim = cfg.pixel_dim
        self.scale_keys = tuple(sorted(in_channels, key=parse_scale))
        self.out_stride = int(cfg.out_stride)
        self.pixel_dim = pixel_dim
        # FPN body stays at pixel_dim (typically 256), not transformer hidden_dim.
        fpn_dim = pixel_dim
        self.fpn_dim = fpn_dim

        self.laterals = nn.ModuleDict()
        self.outputs = nn.ModuleDict()
        for key in self.scale_keys:
            self.laterals[key] = nn.Sequential(
                nn.Conv2d(int(in_channels[key]), fpn_dim, kernel_size=1, bias=False),
                nn.GroupNorm(_gn_groups(fpn_dim), fpn_dim),
            )
            self.outputs[key] = nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(_gn_groups(fpn_dim), fpn_dim),
                nn.ReLU(inplace=True),
            )

        extra = []
        stride = parse_scale(self.scale_keys[0])
        while stride > self.out_stride and stride % 2 == 0:
            extra.append(
                nn.Sequential(
                    nn.ConvTranspose2d(fpn_dim, fpn_dim, kernel_size=2, stride=2),
                    nn.GroupNorm(_gn_groups(fpn_dim), fpn_dim),
                    nn.ReLU(inplace=True),
                )
            )
            stride //= 2
        self.extra_upsample = nn.Sequential(*extra) if extra else nn.Identity()
        self.mask_features = nn.Sequential(
            nn.Conv2d(fpn_dim, pixel_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(pixel_dim), pixel_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        features: Mapping[str, torch.Tensor],
        out_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, FeatureMaps]:
        maps = as_feature_maps(features)
        missing = [key for key in self.scale_keys if key not in maps]
        if missing:
            raise KeyError(f"pixel decoder missing scales {missing}, got {list(maps)}")

        laterals = {key: self.laterals[key](maps[key]) for key in self.scale_keys}
        keys = self.scale_keys
        fpn: Dict[str, torch.Tensor] = {}
        fpn[keys[-1]] = self.outputs[keys[-1]](laterals[keys[-1]])
        for fine_key, coarse_key in zip(reversed(keys[:-1]), reversed(keys[1:])):
            up = F.interpolate(fpn[coarse_key], size=laterals[fine_key].shape[-2:], mode="nearest")
            fpn[fine_key] = self.outputs[fine_key](laterals[fine_key] + up)

        finest = self.scale_keys[0]
        pixel = self.mask_features(self.extra_upsample(fpn[finest]))
        if out_size is not None and tuple(pixel.shape[-2:]) != tuple(out_size):
            pixel = F.interpolate(pixel, size=out_size, mode="bilinear", align_corners=False)
        return pixel, FeatureMaps(fpn, name="fpn", extra=getattr(maps, "extra", {}))
