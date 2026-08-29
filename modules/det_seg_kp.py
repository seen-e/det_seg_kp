"""
Det + Seg + Keypoint model.

Architecture (SAM3 / MaskFormer inspired):
  1. Vision tower extracts multi-scale image features
  2. FPN pixel decoder fuses them into a shared stride-4 pixel feature map
  3. Object query decoder (Deformable DETR) samples multi-scale FPN via MSDeformAttn
  4. Per-query heads predict: class, bbox, instance mask (dot-product), keypoints
"""
from __future__ import annotations

import copy
from typing import Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.layers import Mlp
from torch.nn.init import constant_, xavier_uniform_

from modules.config import ModelConfig
from modules.ops import DeformableDecoderLayer
from modules.utils.common import decode_boxes
from modules.backbone import (
    FeatureMaps,
    as_feature_maps,
    build_vision_tower,
    parse_scale,
    scale_key,
)

# Query decoder attends to FPN levels at least this coarse. Stride-4 stays
# mask / keypoint only — flattening it would be ~16x the current token count.
_MEMORY_MIN_STRIDE = 8


def _gn_groups(num_channels: int, max_groups: int = 32) -> int:
    for groups in (max_groups, 16, 8, 4, 2, 1):
        if num_channels % groups == 0:
            return groups
    return 1


def _tower_channels(tower: nn.Module) -> Dict[str, int]:
    channels = getattr(tower, "num_channels")
    if isinstance(channels, Mapping):
        return {str(k): int(v) for k, v in channels.items()}
    if isinstance(channels, int):
        return {scale_key(int(tower.stride)): int(channels)}
    strides = getattr(tower, "strides", None)
    if strides is None:
        raise TypeError(f"{type(tower).__name__} has no strides to pair with num_channels")
    return {scale_key(int(s)): int(c) for s, c in zip(strides, channels)}


class PixelDecoder(nn.Module):
    """FPN pixel decoder (MaskFormer BasePixelDecoder / Mask2Former style).

    Consumes a :class:`FeatureMaps` dict (e.g. ``{'4x', '8x', '16x'}``).
    Top-down laterals fuse them; the finest map is optionally upsampled to
    ``out_stride`` (default 4) and projected to ``pixel_dim`` for mask / kp heads.

    Returns:
      pixel_features: (B, pixel_dim, H/out_stride, W/out_stride)
      fpn: FeatureMaps of fused maps at each backbone scale
    """

    def __init__(
        self,
        in_channels: Mapping[str, int],
        pixel_dim: int,
        out_stride: int = 4,
    ):
        super().__init__()
        if not in_channels:
            raise ValueError("PixelDecoder needs at least one backbone scale")

        self.scale_keys = tuple(sorted(in_channels, key=parse_scale))
        self.out_stride = int(out_stride)
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


class ObjectQueryDecoder(nn.Module):
    """Deformable-DETR decoder with iterative box refinement.

    Query embedding is 2C and split into:
      - query_pos: learned slot identity, added every layer (not updated)
      - tgt: learned query content, refined by the decoder
    Each layer predicts a box residual, updates the sampling reference
    (detached for the next layer), and is supervised by an auxiliary loss.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        num_queries: int,
        n_levels: int,
        n_points: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_levels = n_levels
        self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        self.reference_points = nn.Linear(hidden_dim, 2)
        self.layers = nn.ModuleList(
            [
                DeformableDecoderLayer(
                    d_model=hidden_dim,
                    n_heads=num_heads,
                    n_levels=n_levels,
                    n_points=n_points,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        bbox_embed = Mlp(hidden_dim, hidden_dim, 4, act_layer=nn.ReLU)
        constant_(bbox_embed.fc2.weight, 0.0)
        constant_(bbox_embed.fc2.bias, 0.0)
        self.bbox_embed = nn.ModuleList(
            copy.deepcopy(bbox_embed) for _ in range(num_layers)
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        xavier_uniform_(self.query_embed.weight)
        constant_(self.reference_points.weight, 0.0)
        constant_(self.reference_points.bias, 0.0)

    def _ref_for_attn(self, reference_points: torch.Tensor) -> torch.Tensor:
        return reference_points[:, :, None, :].expand(-1, -1, self.n_levels, -1)

    def forward(
        self,
        src: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          hs: (num_layers, B, Q, C)
          pred_boxes: (num_layers, B, Q, 4) cxcywh in [0, 1]
        """
        b = src.shape[0]
        query_pos, tgt = torch.split(self.query_embed.weight, self.hidden_dim, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(b, -1, -1)
        tgt = tgt.unsqueeze(0).expand(b, -1, -1)
        reference_points = self.reference_points(query_pos).sigmoid()

        intermediates: List[torch.Tensor] = []
        output_boxes: List[torch.Tensor] = []
        for lid, layer in enumerate(self.layers):
            tgt = layer(
                tgt,
                query_pos,
                self._ref_for_attn(reference_points),
                src,
                spatial_shapes,
                level_start_index,
                src_padding_mask,
            )
            pred_boxes = decode_boxes(self.bbox_embed[lid](tgt), reference_points)
            output_boxes.append(pred_boxes)
            intermediates.append(tgt)
            reference_points = pred_boxes.detach()

        return torch.stack(intermediates), torch.stack(output_boxes)


class DetSegKPHead(nn.Module):
    """Per-query heads for class, mask, and keypoints. Boxes come from the decoder."""

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        pixel_dim: int,
    ):
        super().__init__()
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object
        # Same 2-layer ReLU MLP pattern as decoder bbox_embed (timm.layers.Mlp).
        self.mask_embed = Mlp(hidden_dim, hidden_dim, pixel_dim, act_layer=nn.ReLU)
        self.kp_head = Mlp(hidden_dim, hidden_dim, pixel_dim, act_layer=nn.ReLU)
        self.pixel_dim = pixel_dim

    def forward(
        self,
        obj_queries: torch.Tensor,
        pixel_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        obj_queries: (B, Q, C)
        pixel_features: (B, pixel_dim, H, W) from pixel decoder
        """
        mask_embed = self.mask_embed(obj_queries)
        kp_embed = self.kp_head(obj_queries)
        return {
            "pred_logits": self.class_head(obj_queries),
            "pred_masks": torch.einsum("bqd,bdhw->bqhw", mask_embed, pixel_features),
            "pred_kps": torch.einsum("bqd,bdhw->bqhw", kp_embed, pixel_features),
        }


class DetSegKPModel(nn.Module):
    """
    Unified detection + instance segmentation + keypoint model.

    Pipeline:
      image -> vision_tower -> FPN pixel_decoder -> pixel_features (stride 4)
                            -> multi-scale memory (stride >= 8) -> Deformable-DETR decoder
      obj_queries + pixel_features -> det / mask / kp heads
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.vision_tower = build_vision_tower(cfg.vision_tower)
        in_channels = _tower_channels(self.vision_tower)
        self.pixel_decoder = PixelDecoder(
            in_channels=in_channels,
            pixel_dim=cfg.pixel_dim,
        )
        self.memory_proj = nn.Conv2d(self.pixel_decoder.fpn_dim, cfg.hidden_dim, 1)
        memory_keys = [
            key for key in self.pixel_decoder.scale_keys
            if parse_scale(key) >= _MEMORY_MIN_STRIDE
        ]
        if not memory_keys:
            memory_keys = [self.pixel_decoder.scale_keys[-1]]
        self.memory_keys = tuple(memory_keys)
        self.level_embed = nn.ParameterDict({
            key: nn.Parameter(torch.zeros(cfg.hidden_dim)) for key in self.memory_keys
        })
        for param in self.level_embed.values():
            nn.init.normal_(param)
        self.query_decoder = ObjectQueryDecoder(
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_decoder_layers,
            num_heads=cfg.num_heads,
            num_queries=cfg.num_queries,
            n_levels=len(self.memory_keys),
            n_points=cfg.num_deform_points,
        )
        self.head = DetSegKPHead(
            hidden_dim=cfg.hidden_dim,
            num_classes=cfg.num_classes,
            pixel_dim=cfg.pixel_dim,
        )

    def _build_memory(
        self,
        fpn: Mapping[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flatten FPN levels (stride >= 8) for deformable attention.

        Returns src (B, S, C), spatial_shapes (L, 2), level_start_index (L,).
        Level embeddings are added to content; spatial location is handled by
        reference points + sampling offsets, not sincos mixed into value.
        """
        maps = as_feature_maps(fpn)
        keys = [key for key in self.memory_keys if key in maps]
        if not keys:
            keys = [maps.coarsest_key]

        tokens: List[torch.Tensor] = []
        shapes: List[Tuple[int, int]] = []
        for key in keys:
            proj = self.memory_proj(maps[key])
            _, _, h, w = proj.shape
            embed = self.level_embed[key] if key in self.level_embed else self.level_embed[keys[-1]]
            src = rearrange(proj, "b c h w -> b (h w) c") + embed.to(dtype=proj.dtype)
            tokens.append(src)
            shapes.append((h, w))

        src = torch.cat(tokens, dim=1)
        spatial_shapes = torch.as_tensor(shapes, dtype=torch.long, device=src.device)
        level_start_index = torch.cat((
            spatial_shapes.new_zeros((1,)),
            spatial_shapes.prod(1).cumsum(0)[:-1],
        ))
        return src, spatial_shapes, level_start_index

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = as_feature_maps(self.vision_tower(images))
        out_h = images.shape[-2] // self.pixel_decoder.out_stride
        out_w = images.shape[-1] // self.pixel_decoder.out_stride
        pixel_features, fpn = self.pixel_decoder(features, out_size=(out_h, out_w))

        src, spatial_shapes, level_start_index = self._build_memory(fpn)
        hs, pred_boxes = self.query_decoder(src, spatial_shapes, level_start_index)
        obj_queries = hs[-1]
        outputs = self.head(obj_queries, pixel_features)
        outputs["pred_boxes"] = pred_boxes[-1]
        outputs["aux_outputs"] = [
            {"pred_logits": self.head.class_head(hs[i]), "pred_boxes": pred_boxes[i]}
            for i in range(hs.shape[0] - 1)
        ]
        outputs["pixel_features"] = pixel_features
        outputs["obj_queries"] = obj_queries
        return outputs

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        score_threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> List[Dict[str, torch.Tensor]]:
        """Post-process outputs into per-image predictions."""
        self.eval()
        outputs = self.forward(images)
        return postprocess(outputs, score_threshold, mask_threshold)


def postprocess(
    outputs: Dict[str, torch.Tensor],
    score_threshold: float = 0.5,
    mask_threshold: float = 0.5,
) -> List[Dict[str, torch.Tensor]]:
    """Convert raw model outputs to final predictions per image."""
    pred_logits = outputs["pred_logits"]  # (B, Q, C+1)
    pred_boxes = outputs["pred_boxes"]
    pred_masks = outputs["pred_masks"]
    pred_kps = outputs["pred_kps"]

    prob = F.softmax(pred_logits, dim=-1)
    scores, labels = prob[..., :-1].max(dim=-1)

    results = []
    for b in range(pred_logits.shape[0]):
        keep = scores[b] > score_threshold
        results.append({
            "scores": scores[b][keep],
            "labels": labels[b][keep],
            "boxes": pred_boxes[b][keep],
            "masks": (pred_masks[b][keep].sigmoid() > mask_threshold).float(),
            "kps": pred_kps[b][keep].sigmoid(),
        })
    return results
