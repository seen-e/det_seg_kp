"""Multi-scale deformable attention 2D (Deformable-DETR).

Layout under ``modules/ops/ms_deform_atten_2D``:
  - Python: ``ms_deform_attn_core_Pytorch`` / ``_CUDA`` + ``MSDeformAttn``
  - CUDA: ``src/*.cpp|.cu`` loaded via ``torch.utils.cpp_extension.load``
"""
from __future__ import annotations

import math
import os
import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.nn.init import constant_, xavier_uniform_
from torch.utils.cpp_extension import load

_SRC_DIR = Path(__file__).resolve().parent / "src"
_BUILD_DIR = Path(
    os.environ.get("TORCH_EXTENSIONS_DIR", "/tmp/torch_extensions")
) / "ms_deform_atten"
_BUILD_DIR.mkdir(parents=True, exist_ok=True)

ms_deform_attn_forward_impl = None
ms_deform_attn_backward_impl = None
_MSDA_AVAILABLE = False

if torch.cuda.is_available():
    try:
        ms_deform_atten_cuda = load(
            name="ms_deform_atten",
            sources=[
                str(_SRC_DIR / "ms_deform_atten.cpp"),
                str(_SRC_DIR / "ms_deform_atten.cu"),
            ],
            extra_cflags=["-O2"],
            extra_cuda_cflags=[
                "-O2",
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ],
            build_directory=str(_BUILD_DIR),
            verbose=True,
        )
        ms_deform_attn_forward_impl = ms_deform_atten_cuda.ms_deform_attn_forward
        ms_deform_attn_backward_impl = ms_deform_atten_cuda.ms_deform_attn_backward
        _MSDA_AVAILABLE = True
    except Exception as exc:  # noqa: BLE001 — keep training usable without CUDA op
        warnings.warn(
            f"Failed to load ms_deform_atten CUDA extension ({exc}); "
            "falling back to ms_deform_attn_core_Pytorch (grid_sample).",
            stacklevel=2,
        )


def ms_deform_attn_core_Pytorch(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
) -> torch.Tensor:
    """PyTorch ``grid_sample`` core. value (N,S,M,D), loc (N,Lq,M,L,P,2) -> (N,Lq,M*D)."""
    n, _, n_heads, head_dim = value.shape
    _, n_queries, _, n_levels, n_points, _ = sampling_locations.shape
    del head_dim, n_queries, n_levels, n_points
    split_sizes = [int(h * w) for h, w in spatial_shapes.tolist()]
    grids = 2.0 * sampling_locations - 1.0
    sampled = []
    for level, ((height, width), value_l) in enumerate(
        zip(spatial_shapes.tolist(), value.split(split_sizes, dim=1))
    ):
        value_l = rearrange(
            value_l, "n (h w) m d -> (n m) d h w", h=int(height), w=int(width)
        )
        grid_l = rearrange(grids[:, :, :, level], "n lq m p c -> (n m) lq p c")
        sampled.append(
            F.grid_sample(
                value_l.float(),
                grid_l.float(),
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            ).to(value.dtype)
        )
    weights = rearrange(attention_weights, "n lq m l p -> (n m) 1 lq (l p)")
    out = rearrange(torch.stack(sampled, dim=-2), "nm d lq l p -> nm d lq (l p)")
    out = (out * weights).sum(-1)
    return rearrange(out, "(n m) d lq -> n lq (m d)", n=n, m=n_heads)


class _MSDeformAttnFunction(Function):
    @staticmethod
    def forward(
        ctx,
        value,
        value_spatial_shapes,
        value_level_start_index,
        sampling_locations,
        attention_weights,
        im2col_step,
    ):
        ctx.im2col_step = im2col_step
        output = ms_deform_attn_forward_impl(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            ctx.im2col_step,
        )
        ctx.save_for_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        )
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        (
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
        ) = ctx.saved_tensors
        grad_value, grad_sampling_loc, grad_attn_weight = ms_deform_attn_backward_impl(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output,
            ctx.im2col_step,
        )
        return grad_value, None, None, grad_sampling_loc, grad_attn_weight, None


def ms_deform_attn_core_CUDA(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    level_start_index: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    im2col_step: int = 64,
) -> torch.Tensor:
    """CUDA core via dynamically compiled ``ms_deform_atten`` extension."""
    if not _MSDA_AVAILABLE or ms_deform_attn_forward_impl is None:
        raise RuntimeError("ms_deform_atten CUDA extension is not available")
    if not value.is_cuda:
        raise RuntimeError("ms_deform_attn_core_CUDA requires CUDA tensors")
    dtype = value.dtype
    out = _MSDeformAttnFunction.apply(
        value.float().contiguous(),
        spatial_shapes.contiguous(),
        level_start_index.contiguous(),
        sampling_locations.float().contiguous(),
        attention_weights.float().contiguous(),
        im2col_step,
    )
    return out.to(dtype)


def ms_deform_attn_core(
    value: torch.Tensor,
    spatial_shapes: torch.Tensor,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    level_start_index: Optional[torch.Tensor] = None,
    im2col_step: int = 64,
) -> torch.Tensor:
    """CUDA extension when available on GPU; else PyTorch ``grid_sample`` fallback."""
    use_cuda = _MSDA_AVAILABLE and value.is_cuda and level_start_index is not None
    if use_cuda:
        return ms_deform_attn_core_CUDA(
            value,
            spatial_shapes,
            level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step=im2col_step,
        )
    return ms_deform_attn_core_Pytorch(
        value, spatial_shapes, sampling_locations, attention_weights
    )


class MSDeformAttn(nn.Module):
    """Multi-scale deformable attention (Deformable-DETR).

    Predicts sampling offsets and attention weights from the query; supports
    2D (cx, cy) or 4D (cxcywh) reference points.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
        im2col_step: int = 64,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        self.im2col_step = im2col_step
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()
        if not _MSDA_AVAILABLE:
            warnings.warn(
                "ms_deform_atten CUDA op not available; "
                "MSDeformAttn falls back to ms_deform_attn_core_Pytorch (grid_sample).",
                stacklevel=2,
            )

    def _reset_parameters(self) -> None:
        constant_(self.sampling_offsets.weight, 0.0)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (
            2.0 * math.pi / self.n_heads
        )
        grid = torch.stack([thetas.cos(), thetas.sin()], dim=-1)
        grid = grid / grid.abs().max(dim=-1, keepdim=True).values
        grid = rearrange(grid, "m c -> m 1 1 c")
        grid = grid.repeat(1, self.n_levels, self.n_points, 1)
        for p in range(self.n_points):
            grid[:, :, p] *= p + 1
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(rearrange(grid, "... -> (...)"))
        constant_(self.attention_weights.weight, 0.0)
        constant_(self.attention_weights.bias, 0.0)
        xavier_uniform_(self.value_proj.weight)
        xavier_uniform_(self.output_proj.weight)
        constant_(self.value_proj.bias, 0.0)
        constant_(self.output_proj.bias, 0.0)

    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        src: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        n_src = src.shape[1]
        if int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum()) != n_src:
            raise ValueError("spatial_shapes do not match flattened src length")

        value = self.value_proj(src)
        if padding_mask is not None:
            value = value.masked_fill(padding_mask[..., None], 0.0)
        value = rearrange(value, "n s (m d) -> n s m d", m=self.n_heads)
        offsets = rearrange(
            self.sampling_offsets(query),
            "n lq (m lvl p two) -> n lq m lvl p two",
            m=self.n_heads,
            lvl=self.n_levels,
            p=self.n_points,
            two=2,
        )
        attn = F.softmax(
            rearrange(
                self.attention_weights(query),
                "n lq (m lp) -> n lq m lp",
                m=self.n_heads,
            ),
            dim=-1,
        )
        attn = rearrange(
            attn,
            "n lq m (lvl p) -> n lq m lvl p",
            lvl=self.n_levels,
            p=self.n_points,
        )

        ref = reference_points[:, :, None, :, None, :]
        if reference_points.shape[-1] == 2:
            hw = torch.stack(
                [spatial_shapes[:, 1], spatial_shapes[:, 0]], dim=-1
            ).to(offsets.dtype)
            locations = ref + offsets / hw[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            locations = ref[..., :2] + offsets / self.n_points * ref[..., 2:] * 0.5
        else:
            raise ValueError(
                f"reference_points last dim must be 2 or 4, got {reference_points.shape[-1]}"
            )
        return self.output_proj(
            ms_deform_attn_core(
                value,
                spatial_shapes,
                locations,
                attn,
                level_start_index=level_start_index,
                im2col_step=self.im2col_step,
            )
        )
