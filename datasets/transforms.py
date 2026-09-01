"""Geometric and photometric augmentation for image / instance-mask / kps.

Geometric ops (scale / crop / hflip) are composed into one affine and applied
once via ``cv2.warpAffine``. Color jitter is image-only. Instance filters run
after geometry (labels stay in sync).

Box corner order (from gen_train_sample): bottom 0-3 CCW, top 4-7 CCW,
local +X = width. Horizontal flip swaps left/right corners:
0↔1, 3↔2, 4↔5, 7↔6.
"""
from __future__ import annotations

import random
from typing import Tuple

import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import ColorJitter

from modules.config import DataConfig
from scripts.gen_train_sample import _KP_INVALID, ensure_kps_xyv

from .filter import ComposeInstanceFilters, build_instance_filters

# After a left-right flip, remap corner indices so semantic identity is kept.
_FLIP_CORNER_PERM = np.array([1, 0, 3, 2, 5, 4, 7, 6], dtype=np.int64)

_MASK_BG = 255


def invalidate_oob_kps(kps: np.ndarray, width: int, height: int) -> np.ndarray:
    """Mark corners outside [0, W) x [0, H) as invalid (v=1)."""
    kps = ensure_kps_xyv(kps)
    if kps.size == 0:
        return kps
    x, y = kps[..., 0], kps[..., 1]
    oob = (x < 0.0) | (x >= float(width)) | (y < 0.0) | (y >= float(height))
    kps = kps.copy()
    kps[..., 2] = np.where(oob, _KP_INVALID, kps[..., 2])
    return kps


def _eye3() -> np.ndarray:
    return np.eye(3, dtype=np.float64)


def _scale_mat(sx: float, sy: float) -> np.ndarray:
    M = _eye3()
    M[0, 0] = float(sx)
    M[1, 1] = float(sy)
    return M


def _translate_mat(tx: float, ty: float) -> np.ndarray:
    M = _eye3()
    M[0, 2] = float(tx)
    M[1, 2] = float(ty)
    return M


def _hflip_mat(width: int) -> np.ndarray:
    """x' = (W - 1) - x  (matches previous PIL flip convention)."""
    M = _eye3()
    M[0, 0] = -1.0
    M[0, 2] = float(width - 1)
    return M


def _rotate_mat(angle_deg: float, cx: float, cy: float) -> np.ndarray:
    """Rotate CCW around (cx, cy); ready for future aug, unused by default."""
    th = np.deg2rad(float(angle_deg))
    c, s = np.cos(th), np.sin(th)
    R = _eye3()
    R[0, 0], R[0, 1], R[1, 0], R[1, 1] = c, -s, s, c
    return _translate_mat(cx, cy) @ R @ _translate_mat(-cx, -cy)


def _as_2x3(M3: np.ndarray) -> np.ndarray:
    return np.asarray(M3[:2, :], dtype=np.float64)


def _apply_affine_xy(xy: np.ndarray, M_fwd: np.ndarray) -> np.ndarray:
    """Apply 3x3 forward affine to (..., 2) points."""
    x = xy[..., 0]
    y = xy[..., 1]
    return np.stack(
        [
            M_fwd[0, 0] * x + M_fwd[0, 1] * y + M_fwd[0, 2],
            M_fwd[1, 0] * x + M_fwd[1, 1] * y + M_fwd[1, 2],
        ],
        axis=-1,
    )


def _clip_boxes_xyxy(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clip exclusive xyxy to ``[0, W] x [0, H]``; zero out empty boxes."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    boxes[:, 0] = np.clip(boxes[:, 0], 0.0, float(width))
    boxes[:, 2] = np.clip(boxes[:, 2], 0.0, float(width))
    boxes[:, 1] = np.clip(boxes[:, 1], 0.0, float(height))
    boxes[:, 3] = np.clip(boxes[:, 3], 0.0, float(height))
    empty = ~((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))
    boxes[empty] = 0.0
    return boxes


def _transform_boxes_xyxy(boxes: np.ndarray, M_fwd: np.ndarray) -> np.ndarray:
    """Map exclusive xyxy through affine; rebuild axis-aligned AABB."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if boxes.size == 0:
        return boxes.copy()
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    corners = np.stack(
        [
            np.stack([x0, y0], axis=-1),
            np.stack([x1, y0], axis=-1),
            np.stack([x1, y1], axis=-1),
            np.stack([x0, y1], axis=-1),
        ],
        axis=1,
    )  # (N, 4, 2)
    mapped = _apply_affine_xy(corners, M_fwd)
    out = np.zeros_like(boxes)
    out[:, 0] = mapped[:, :, 0].min(axis=1)
    out[:, 1] = mapped[:, :, 1].min(axis=1)
    out[:, 2] = mapped[:, :, 0].max(axis=1)
    out[:, 3] = mapped[:, :, 1].max(axis=1)
    empty = ~((boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1]))
    out[empty] = 0.0
    return out


def _warp_image_mask_kps(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    kps: np.ndarray,
    boxes: np.ndarray,
    M_fwd: np.ndarray,
    out_w: int,
    out_h: int,
    *,
    do_flip_perm: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One ``warpAffine`` for image + mask; transform kps/boxes with the same M.

    ``M_fwd`` maps source → output. OpenCV ``warpAffine`` (without
    ``WARP_INVERSE_MAP``) treats ``M`` as that forward map and inverts it
    internally for sampling — pass ``M_fwd``, not ``inv(M_fwd)``.
    """
    M_2x3 = _as_2x3(M_fwd)
    image_out = cv2.warpAffine(
        np.ascontiguousarray(image_rgb),
        M_2x3,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    mask_out = cv2.warpAffine(
        np.ascontiguousarray(mask),
        M_2x3,
        (out_w, out_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=int(_MASK_BG),
    )
    kps = ensure_kps_xyv(kps).copy()
    if kps.size:
        kps[..., :2] = _apply_affine_xy(kps[..., :2], M_fwd)
        if do_flip_perm:
            kps = kps[:, _FLIP_CORNER_PERM]
    kps = invalidate_oob_kps(kps, out_w, out_h)
    boxes_out = _clip_boxes_xyxy(_transform_boxes_xyxy(boxes, M_fwd), out_w, out_h)
    return image_out, mask_out, kps, boxes_out


class DetSegKPTransform:
    """Scale+crop (+ optional hflip) via one affine warp, then color / filters.

    Train: keep aspect ratio, scale so the image covers
    ``(img_width, img_height) * Uniform(scale_min, scale_max)``, then random
    crop to ``img_width x img_height``, flip, ColorJitter, instance filters.

    Val / no-aug: scale to cover the canvas, center crop, no flip / color;
    same instance filters as train.
    Native 1920x1536 → 960x768 is an exact 2x downsample (no crop leftover).
    """

    def __init__(self, cfg: DataConfig, train: bool = True):
        self.img_w = int(cfg.img_width)
        self.img_h = int(cfg.img_height)
        self.stride = int(cfg.stride)
        self.train = bool(train)
        self.hflip_prob = float(cfg.hflip_prob) if train else 0.0
        self.scale_min = float(cfg.scale_min)
        self.scale_max = float(cfg.scale_max)
        self.instance_filters: ComposeInstanceFilters = build_instance_filters(
            cfg.min_box_side_hm,
            cfg.min_box_area_hm,
            cfg.min_mask_pixels,
        )
        if train:
            self.color_jitter = ColorJitter(
                brightness=cfg.color_brightness,
                contrast=cfg.color_contrast,
                saturation=cfg.color_saturation,
                hue=cfg.color_hue,
            )
        else:
            self.color_jitter = None

    def _scale_factor(self) -> float:
        if not self.train:
            return 1.0
        return random.uniform(self.scale_min, self.scale_max)

    def _sample_geometry(
        self, src_w: int, src_h: int
    ) -> Tuple[np.ndarray, bool]:
        """Build forward affine (src → out) and whether hflip corner-perm is needed."""
        cover = max(self.img_w / float(src_w), self.img_h / float(src_h))
        factor = cover * self._scale_factor()
        new_w = max(int(round(src_w * factor)), 1)
        new_h = max(int(round(src_h * factor)), 1)
        sx = new_w / float(src_w)
        sy = new_h / float(src_h)

        # Same crop window as old pad-then-crop (pad is implicit border in warp).
        padded_w = max(new_w, self.img_w)
        padded_h = max(new_h, self.img_h)
        max_left = padded_w - self.img_w
        max_top = padded_h - self.img_h
        if self.train:
            left = random.randint(0, max_left) if max_left > 0 else 0
            top = random.randint(0, max_top) if max_top > 0 else 0
        else:
            left = max_left // 2
            top = max_top // 2

        do_flip = self.train and random.random() < self.hflip_prob
        # src → scaled → crop window → optional hflip
        M = _translate_mat(-left, -top) @ _scale_mat(sx, sy)
        if do_flip:
            M = _hflip_mat(self.img_w) @ M
        return M, do_flip

    def __call__(
        self,
        image: Image.Image,
        mask: np.ndarray,
        kps: np.ndarray,
        labels: np.ndarray,
        boxes: np.ndarray,
    ) -> Tuple[Image.Image, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if image.mode != "RGB":
            image = image.convert("RGB")
        image_rgb = np.asarray(image, dtype=np.uint8)
        mask = np.asarray(mask, dtype=np.uint8)
        if mask.ndim == 3:
            mask = mask[..., 0]
        src_h, src_w = image_rgb.shape[:2]
        kps = ensure_kps_xyv(kps)
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)

        M_fwd, do_flip = self._sample_geometry(src_w, src_h)
        image_rgb, mask, kps, boxes = _warp_image_mask_kps(
            image_rgb,
            mask,
            kps,
            boxes,
            M_fwd,
            self.img_w,
            self.img_h,
            do_flip_perm=do_flip,
        )
        image = Image.fromarray(image_rgb)

        if self.color_jitter is not None:
            image = self.color_jitter(image)
        labels = np.asarray(labels, dtype=np.int64)
        mask, kps, labels, boxes = self.instance_filters(
            mask, kps, labels, boxes, stride=self.stride
        )
        return image, mask, kps, labels, boxes
