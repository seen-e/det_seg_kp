"""Deformable-DETR decoder layer: self-attn + MSDeformAttn + FFN."""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from timm.layers import Mlp

from .ms_deform_atten_2D import MSDeformAttn


class DeformableDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_levels: int,
        n_points: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.ffn = Mlp(
            d_model, dim_feedforward, d_model, act_layer=nn.ReLU, drop=(dropout, 0.0)
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList(nn.LayerNorm(d_model) for _ in range(3))

    def _residual(self, x: torch.Tensor, y: torch.Tensor, i: int) -> torch.Tensor:
        return self.norm[i](x + self.dropout(y))

    def forward(
        self,
        tgt: torch.Tensor,
        query_pos: torch.Tensor,
        reference_points: torch.Tensor,
        src: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = k = tgt + query_pos
        tgt = self._residual(tgt, self.self_attn(q, k, tgt, need_weights=False)[0], 0)
        tgt = self._residual(
            tgt,
            self.cross_attn(
                tgt + query_pos,
                reference_points,
                src,
                spatial_shapes,
                level_start_index,
                src_padding_mask,
            ),
            1,
        )
        tgt = self._residual(tgt, self.ffn(tgt), 2)
        return tgt
