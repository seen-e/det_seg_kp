"""Deformable-DETR object query decoder with iterative box refinement."""
from __future__ import annotations

import copy
from typing import List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import Mlp
from torch.nn.init import constant_, xavier_uniform_

from ..backbone import as_feature_maps, parse_scale
from ..config import ObjectQueryDecoderConfig
from ..ops import DeformableDecoderLayer
from ..utils.common import decode_boxes


def _select_memory_keys(scale_keys: List[str], memory_min_stride: int) -> Tuple[str, ...]:
    keys = [key for key in scale_keys if parse_scale(key) >= memory_min_stride]
    if keys:
        return tuple(keys)
    if not scale_keys:
        raise ValueError("DeformDETRObjectDecoder needs at least one FPN scale_key")
    return (scale_keys[-1],)


class DeformDETRObjectDecoder(nn.Module):
    """Deformable-DETR decoder with iterative box refinement.

    Owns FPN → multi-scale memory projection (``memory_proj`` + level
    embeddings), then runs the deformable decoder.

    Query embedding is 2C and split into:
      - query_pos: learned slot identity, added every layer (not updated)
      - tgt: learned query content, refined by the decoder

    Reference starts as 2D centers (cx, cy). Each layer predicts a box residual;
    detached pred boxes (cxcywh) become the next-layer reference and are
    supervised by an auxiliary loss.
    """

    def __init__(self, cfg: ObjectQueryDecoderConfig):
        super().__init__()
        hidden_dim = cfg.hidden_dim
        self.hidden_dim = hidden_dim
        self.num_layers = cfg.num_layers

        scale_keys = list(cfg.scale_keys)
        self.memory_keys = _select_memory_keys(scale_keys, cfg.memory_min_stride)
        self.n_levels = len(self.memory_keys)

        self.memory_proj = nn.Conv2d(cfg.fpn_dim, hidden_dim, 1)
        self.level_embed = nn.ParameterDict({
            key: nn.Parameter(torch.zeros(hidden_dim)) for key in self.memory_keys
        })
        for param in self.level_embed.values():
            nn.init.normal_(param)

        self.query_embed = nn.Embedding(cfg.num_queries, hidden_dim * 2)
        self.reference_points = nn.Linear(hidden_dim, 2)
        layer = DeformableDecoderLayer(
            d_model=hidden_dim,
            n_heads=cfg.num_heads,
            n_levels=self.n_levels,
            n_points=cfg.n_points,
            dim_feedforward=hidden_dim * 4,
            dropout=cfg.dropout,
        )
        self.layers = nn.ModuleList(
            copy.deepcopy(layer) for _ in range(cfg.num_layers)
        )
        bbox_embed = Mlp(hidden_dim, hidden_dim, 4, act_layer=nn.ReLU)
        constant_(bbox_embed.fc2.weight, 0.0)
        constant_(bbox_embed.fc2.bias, 0.0)
        self.bbox_embed = nn.ModuleList(
            copy.deepcopy(bbox_embed) for _ in range(cfg.num_layers)
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        xavier_uniform_(self.query_embed.weight)
        constant_(self.reference_points.weight, 0.0)
        constant_(self.reference_points.bias, 0.0)

    def _ref_for_attn(self, reference_points: torch.Tensor) -> torch.Tensor:
        return reference_points[:, :, None, :].expand(-1, -1, self.n_levels, -1)

    def _build_memory(
        self,
        fpn: Mapping[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flatten FPN levels for deformable attention.

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

    def forward(
        self,
        fpn: Mapping[str, torch.Tensor],
        src_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          fpn: multi-scale FPN feature maps from the pixel decoder

        Returns:
          hs: (num_layers, B, Q, C)
          pred_boxes: (num_layers, B, Q, 4) cxcywh in [0, 1]
        """
        src, spatial_shapes, level_start_index = self._build_memory(fpn)
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
