"""Composable instance filters applied after geometric transforms.

Thresholds for side / area are specified in *heatmap* space and scaled by
``stride`` / ``stride**2`` so a larger stride filters more aggressively.
``min_mask_pixels`` is counted on the downsampled id-mask (already stride-aware).

When ``boxes`` (exclusive xyxy, aligned with instance ids) are provided, side /
area filters use them and skip scanning the full-resolution mask.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from scripts.gen_train_sample import ensure_kps_xyv


_BG = 255


def _instance_aabbs(mask: np.ndarray, num_instances: int) -> np.ndarray:
    """Per-instance AABB in input pixels: (N, 4) as x0, y0, x1, y1 (x1/y1 exclusive)."""
    from scripts.gen_train_sample import mask_id_to_boxes_xyxy

    return mask_id_to_boxes_xyxy(mask, num_instances, exclusive=True)


def _downsample_id_mask(mask: np.ndarray, stride: int) -> np.ndarray:
    h, w = mask.shape
    out_h, out_w = h // stride, w // stride
    if out_h < 1 or out_w < 1:
        raise ValueError(f"mask {h}x{w} is smaller than stride={stride}")
    return np.asarray(
        Image.fromarray(mask).resize((out_w, out_h), Image.NEAREST),
        dtype=np.uint8,
    )


def apply_keep(
    mask: np.ndarray,
    kps: np.ndarray,
    labels: np.ndarray,
    keep: np.ndarray,
    boxes: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Drop instances where ``keep`` is False; remap remaining ids to 0..K-1."""
    kps = ensure_kps_xyv(kps)
    labels = np.asarray(labels, dtype=np.int64)
    keep = np.asarray(keep, dtype=bool)
    boxes_arr = None if boxes is None else np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

    old_ids = np.nonzero(keep)[0]
    if old_ids.size == 0:
        out_mask = np.full(mask.shape, _BG, dtype=np.uint8)
        return (
            out_mask,
            np.zeros((0, 8, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 4), dtype=np.float32) if boxes_arr is not None else None,
        )

    lut = np.full(256, _BG, dtype=np.uint8)
    for new_id, old_id in enumerate(old_ids):
        lut[int(old_id)] = np.uint8(new_id)
    out_mask = lut[mask]
    out_boxes = boxes_arr[keep] if boxes_arr is not None else None
    return out_mask, kps[keep], labels[keep], out_boxes


class InstanceFilter(ABC):
    """Boolean keep-mask over instances aligned with id-mask labels 0..N-1."""

    @abstractmethod
    def keep(
        self,
        mask: np.ndarray,
        kps: np.ndarray,
        labels: np.ndarray,
        *,
        stride: int,
        boxes: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return bool array of shape (N,)."""


class MinBoxSideFilter(InstanceFilter):
    """Keep instances whose AABB short side >= ``min_side_hm * stride`` (input px)."""

    def __init__(self, min_side_hm: float):
        self.min_side_hm = float(min_side_hm)

    def keep(
        self,
        mask: np.ndarray,
        kps: np.ndarray,
        labels: np.ndarray,
        *,
        stride: int,
        boxes: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n = int(np.asarray(labels).shape[0])
        if n == 0 or self.min_side_hm <= 0:
            return np.ones((n,), dtype=bool)
        if boxes is None:
            boxes = _instance_aabbs(mask, n)
        else:
            boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        thresh = self.min_side_hm * float(stride)
        return np.minimum(w, h) >= thresh


class MinBoxAreaFilter(InstanceFilter):
    """Keep instances whose AABB area >= ``min_area_hm * stride**2`` (input px^2)."""

    def __init__(self, min_area_hm: float):
        self.min_area_hm = float(min_area_hm)

    def keep(
        self,
        mask: np.ndarray,
        kps: np.ndarray,
        labels: np.ndarray,
        *,
        stride: int,
        boxes: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        n = int(np.asarray(labels).shape[0])
        if n == 0 or self.min_area_hm <= 0:
            return np.ones((n,), dtype=bool)
        if boxes is None:
            boxes = _instance_aabbs(mask, n)
        else:
            boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        thresh = self.min_area_hm * float(stride) ** 2
        return area >= thresh


class MinMaskPixelsFilter(InstanceFilter):
    """Keep instances with >= ``min_pixels`` foreground pixels after stride downsample."""

    def __init__(self, min_pixels: int):
        self.min_pixels = int(min_pixels)

    def keep(
        self,
        mask: np.ndarray,
        kps: np.ndarray,
        labels: np.ndarray,
        *,
        stride: int,
        boxes: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        del boxes  # unused; counts come from the mask
        n = int(np.asarray(labels).shape[0])
        if n == 0 or self.min_pixels <= 0:
            return np.ones((n,), dtype=bool)
        mask_ds = _downsample_id_mask(mask, stride)
        flat = mask_ds.ravel()
        valid = flat < n
        counts = np.bincount(flat[valid], minlength=n)[:n]
        return counts >= self.min_pixels


class ComposeInstanceFilters:
    """AND-compose filters, then remap mask / kps / labels / boxes."""

    def __init__(self, filters: Optional[Sequence[InstanceFilter]] = None):
        self.filters: List[InstanceFilter] = list(filters or [])

    def __call__(
        self,
        mask: np.ndarray,
        kps: np.ndarray,
        labels: np.ndarray,
        boxes: Optional[np.ndarray] = None,
        *,
        stride: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        labels = np.asarray(labels, dtype=np.int64)
        kps = ensure_kps_xyv(kps)
        n = int(labels.shape[0])
        boxes_arr = None if boxes is None else np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

        keep = np.ones((n,), dtype=bool)
        for f in self.filters:
            keep &= f.keep(mask, kps, labels, stride=stride, boxes=boxes_arr)
        return apply_keep(mask, kps, labels, keep, boxes_arr)


def build_instance_filters(
    min_box_side_hm: float,
    min_box_area_hm: float,
    min_mask_pixels: int,
) -> ComposeInstanceFilters:
    """Build compose from config; thresholds <= 0 omit that filter."""
    filters: List[InstanceFilter] = []
    if min_box_side_hm > 0:
        filters.append(MinBoxSideFilter(min_box_side_hm))
    if min_box_area_hm > 0:
        filters.append(MinBoxAreaFilter(min_box_area_hm))
    if min_mask_pixels > 0:
        filters.append(MinMaskPixelsFilter(min_mask_pixels))
    return ComposeInstanceFilters(filters)
