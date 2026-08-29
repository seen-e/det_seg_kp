"""Geometric and photometric augmentation for image / instance-mask / kps.

All geometric ops keep the three aligned. Color jitter is image-only.

Box corner order (from gen_train_sample): bottom 0-3 CCW, top 4-7 CCW,
local +X = width. Horizontal flip swaps left/right corners:
0↔1, 3↔2, 4↔5, 7↔6.
"""
from __future__ import annotations

import random
from typing import Tuple

import numpy as np
from PIL import Image
from torchvision.transforms import ColorJitter

from modules.config import DataConfig
from scripts.gen_train_sample import _KP_INVALID, ensure_kps_xyv

# After a left-right flip, remap corner indices so semantic identity is kept.
_FLIP_CORNER_PERM = np.array([1, 0, 3, 2, 5, 4, 7, 6], dtype=np.int64)


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


def _resize(
    image: Image.Image,
    mask: np.ndarray,
    kps: np.ndarray,
    new_w: int,
    new_h: int,
) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
    old_w, old_h = image.size
    image = image.resize((new_w, new_h), Image.BILINEAR)
    mask_img = Image.fromarray(mask).resize((new_w, new_h), Image.NEAREST)
    mask = np.asarray(mask_img, dtype=np.uint8)
    kps = ensure_kps_xyv(kps)
    if kps.size:
        kps = kps.copy()
        kps[..., 0] *= new_w / float(old_w)
        kps[..., 1] *= new_h / float(old_h)
    return image, mask, kps


def _pad_to_min(
    image: Image.Image,
    mask: np.ndarray,
    kps: np.ndarray,
    min_w: int,
    min_h: int,
    fill: int = 0,
    mask_fill: int = 255,
) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
    w, h = image.size
    pad_w = max(min_w - w, 0)
    pad_h = max(min_h - h, 0)
    if pad_w == 0 and pad_h == 0:
        return image, mask, kps
    canvas = Image.new("RGB", (w + pad_w, h + pad_h), (fill, fill, fill))
    canvas.paste(image, (0, 0))
    padded = np.full((h + pad_h, w + pad_w), mask_fill, dtype=np.uint8)
    padded[:h, :w] = mask
    return canvas, padded, kps


def _crop(
    image: Image.Image,
    mask: np.ndarray,
    kps: np.ndarray,
    left: int,
    top: int,
    crop_w: int,
    crop_h: int,
) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
    image = image.crop((left, top, left + crop_w, top + crop_h))
    mask = np.ascontiguousarray(mask[top : top + crop_h, left : left + crop_w])
    kps = ensure_kps_xyv(kps)
    if kps.size:
        kps = kps.copy()
        kps[..., 0] -= float(left)
        kps[..., 1] -= float(top)
    return image, mask, invalidate_oob_kps(kps, crop_w, crop_h)


def _hflip(
    image: Image.Image,
    mask: np.ndarray,
    kps: np.ndarray,
) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
    w, h = image.size
    image = image.transpose(Image.FLIP_LEFT_RIGHT)
    mask = np.ascontiguousarray(mask[:, ::-1])
    kps = ensure_kps_xyv(kps)
    if kps.size:
        kps = kps.copy()
        kps[..., 0] = float(w - 1) - kps[..., 0]
        kps = kps[:, _FLIP_CORNER_PERM]
    return image, mask, invalidate_oob_kps(kps, w, h)


class DetSegKPTransform:
    """Resize+crop, optional left-right flip, optional color jitter.

    Train: keep aspect ratio, scale so the image covers
    ``(img_width, img_height) * Uniform(scale_min, scale_max)``, then random
    crop to ``img_width x img_height``, flip, ColorJitter.

    Val / no-aug: scale to cover the canvas, center crop, no flip / color.
    Native 1920x1536 → 960x768 is an exact 2x downsample (no crop leftover).
    """

    def __init__(self, cfg: DataConfig, train: bool = True):
        self.img_w = int(cfg.img_width)
        self.img_h = int(cfg.img_height)
        self.train = bool(train)
        self.hflip_prob = float(cfg.hflip_prob) if train else 0.0
        self.scale_min = float(cfg.scale_min)
        self.scale_max = float(cfg.scale_max)
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

    def _resize_and_crop(
        self,
        image: Image.Image,
        mask: np.ndarray,
        kps: np.ndarray,
    ) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
        w, h = image.size
        cover = max(self.img_w / float(w), self.img_h / float(h))
        factor = cover * self._scale_factor()
        new_w = max(int(round(w * factor)), 1)
        new_h = max(int(round(h * factor)), 1)
        image, mask, kps = _resize(image, mask, kps, new_w, new_h)
        image, mask, kps = _pad_to_min(image, mask, kps, self.img_w, self.img_h)

        w, h = image.size
        max_left = w - self.img_w
        max_top = h - self.img_h
        if self.train:
            left = random.randint(0, max_left) if max_left > 0 else 0
            top = random.randint(0, max_top) if max_top > 0 else 0
        else:
            left = max_left // 2
            top = max_top // 2
        return _crop(image, mask, kps, left, top, self.img_w, self.img_h)

    def __call__(
        self,
        image: Image.Image,
        mask: np.ndarray,
        kps: np.ndarray,
    ) -> Tuple[Image.Image, np.ndarray, np.ndarray]:
        image, mask, kps = self._resize_and_crop(image, mask, kps)
        if self.train and random.random() < self.hflip_prob:
            image, mask, kps = _hflip(image, mask, kps)
        if self.color_jitter is not None:
            image = self.color_jitter(image)
        return image, mask, kps
