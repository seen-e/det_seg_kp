"""Dataset and dataloader for det + seg + keypoint.

Each on-disk sample (stem e.g. ``frame_0001``)::

    {data_root}/images/{stem}.png
    {data_root}/labels/{stem}_instance_mask.png
    {data_root}/labels/{stem}_train.json

``__getitem__`` returns training tensors after resize+crop / flip / color jitter:
  - image:   (3, H, W) RGB in [0, 1], W=img_width, H=img_height (default 960x768)
  - masks:   (N, Hs, Ws) binary instance masks
  - kp_maps: (N, Hs, Ws) Gaussian heatmap of that instance's valid corners
  - labels, boxes, kps (kps stay in input-image pixels)

Hs = H / stride, Ws = W / stride (see ``DataConfig.stride``, default 4).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .transforms import DetSegKPTransform
from modules.config import Config, DataConfig
from scripts.gen_train_sample import (
    TRAIN_LABEL_SUFFIX,
    ensure_kps_xyv,
    kp_corner_valid,
    kp_xy,
    load_train_sample_from_labels,
)
from modules.utils import gaussian_heatmap


def instance_mask_to_binary(mask: np.ndarray, num_instances: int) -> np.ndarray:
    """
    Convert an id map (H, W) with labels 0..k-1 / bg=255 into (k, H, W) float32
    binary masks used by the mask / dice losses.
    """
    h, w = mask.shape
    if num_instances <= 0:
        return np.zeros((0, h, w), dtype=np.float32)
    binary = np.zeros((num_instances, h, w), dtype=np.float32)
    for i in range(num_instances):
        binary[i] = (mask == i).astype(np.float32)
    return binary


def kps_to_heatmap(
    kps: np.ndarray,
    height: int,
    width: int,
    sigma: float = 2.0,
) -> np.ndarray:
    """
    Draw up to 8 corners onto a single heatmap.

    Args:
        kps: (8, 3) or (8, 2) legacy; a point is skipped unless v == 0.
             Coordinates are in this heatmap's pixel space.
    Returns:
        heatmap: (H, W) float32 in [0, 1]
    """
    heatmap = np.zeros((height, width), dtype=np.float32)
    kps = np.asarray(kps, dtype=np.float32)
    if kps.ndim == 1:
        kps = rearrange(kps, "c -> 1 c")
    for kp in kps:
        if not kp_corner_valid(kp):
            continue
        x, y = kp_xy(kp)
        blob = gaussian_heatmap((x, y), (height, width), sigma=sigma)
        np.maximum(heatmap, blob, out=heatmap)
    return heatmap


def kps_to_heatmaps(
    kps: np.ndarray,
    height: int,
    width: int,
    sigma: float = 2.0,
) -> np.ndarray:
    """(k, 8, 3) -> (k, H, W); one fused heatmap per instance."""
    kps = np.asarray(kps, dtype=np.float32)
    if kps.size == 0:
        return np.zeros((0, height, width), dtype=np.float32)
    if kps.ndim == 2:
        kps = rearrange(kps, "(k n) c -> k n c", n=8)
    else:
        kps = rearrange(kps, "k n c -> k n c", n=8)
    return np.stack(
        [kps_to_heatmap(kps[i], height, width, sigma=sigma) for i in range(kps.shape[0])],
        axis=0,
    )


def masks_to_boxes_cxcywh(masks: np.ndarray) -> np.ndarray:
    """Tight AABB of each binary mask, normalized cxcywh in [0, 1]."""
    n, h, w = masks.shape
    boxes = np.zeros((n, 4), dtype=np.float32)
    for i in range(n):
        ys, xs = np.where(masks[i] > 0)
        if xs.size == 0:
            boxes[i] = (0.5, 0.5, 0.1, 0.1)
            continue
        x0 = float(xs.min())
        y0 = float(ys.min())
        x1 = float(xs.max()) + 1.0
        y1 = float(ys.max()) + 1.0
        boxes[i, 0] = 0.5 * (x0 + x1) / w
        boxes[i, 1] = 0.5 * (y0 + y1) / h
        boxes[i, 2] = (x1 - x0) / w
        boxes[i, 3] = (y1 - y0) / h
    return boxes


def downsample_id_mask(mask: np.ndarray, stride: int) -> np.ndarray:
    """Nearest-neighbor downsample of an instance-id map to H/stride, W/stride."""
    h, w = mask.shape
    out_h, out_w = h // stride, w // stride
    if out_h < 1 or out_w < 1:
        raise ValueError(f"mask {h}x{w} is smaller than stride={stride}")
    return np.asarray(
        Image.fromarray(mask).resize((out_w, out_h), Image.NEAREST),
        dtype=np.uint8,
    )


def discover_stems(data_root: Path) -> List[str]:
    """List sample stems from ``labels/index.json`` or ``*_train.json`` files."""
    labels_dir = data_root / "labels"
    index_path = labels_dir / "index.json"
    if index_path.is_file():
        with index_path.open(encoding="utf-8") as f:
            payload = json.load(f)
        samples = payload.get("samples", payload if isinstance(payload, list) else [])
        stems = [str(e["stem"]) for e in samples]
        if stems:
            return stems
    paths = sorted(labels_dir.glob(f"*{TRAIN_LABEL_SUFFIX}.json"))
    suffix = f"{TRAIN_LABEL_SUFFIX}.json"
    return [p.name[: -len(suffix)] for p in paths]


def split_stems(
    stems: Sequence[str],
    split: str,
    val_ratio: float,
    seed: int,
) -> List[str]:
    """Deterministic train/val split. ``val_ratio <= 0`` uses every stem for both."""
    stems = list(stems)
    if not stems:
        return []
    if val_ratio <= 0.0:
        return stems
    rng = np.random.RandomState(int(seed))
    order = rng.permutation(len(stems))
    shuffled = [stems[i] for i in order]
    n_val = max(1, int(round(len(shuffled) * float(val_ratio))))
    n_val = min(n_val, len(shuffled) - 1) if len(shuffled) > 1 else n_val
    val = set(shuffled[:n_val])
    if split == "val":
        return [s for s in stems if s in val]
    return [s for s in stems if s not in val]


def _worker_init_fn(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(seed)
    random.seed(seed)


class DetSegKPDataset(Dataset):
    """
    File-backed samples with train-time augmentation.

    Returns a dict:
      image:   (3, H, W) float32 RGB in [0, 1]
      masks:   (N, Hs, Ws) binary instance masks, Hs=H/stride
      kp_maps: (N, Hs, Ws) fused keypoint heatmaps
      labels:  (N,) class ids
      boxes:   (N, 4) normalized cxcywh (from full-res silhouette after aug)
      kps:     (N, 8, 3) pixel corners on the input image; v=0 valid, v=1 invalid
      stem:    sample id (str)
    """

    def __init__(
        self,
        cfg: Optional[DataConfig] = None,
        split: str = "train",
        augment: Optional[bool] = None,
        data_root: Optional[str] = None,
    ):
        self.cfg = cfg if cfg is not None else DataConfig()
        if data_root is not None:
            self.cfg.data_root = data_root
        self.split = split
        self.img_w = int(self.cfg.img_width)
        self.img_h = int(self.cfg.img_height)
        self.stride = int(self.cfg.stride)
        self.num_kps = int(self.cfg.num_kps)
        self.kp_sigma = float(self.cfg.kp_sigma)
        if self.img_w % self.stride != 0 or self.img_h % self.stride != 0:
            raise ValueError(
                f"img_width={self.img_w}, img_height={self.img_h} must be "
                f"divisible by stride={self.stride}"
            )
        self.out_w = self.img_w // self.stride
        self.out_h = self.img_h // self.stride

        self.data_root = Path(self.cfg.data_root)
        self.images_dir = self.data_root / "images"
        self.labels_dir = self.data_root / "labels"
        stems = discover_stems(self.data_root)
        if not stems:
            raise FileNotFoundError(
                f"No training samples under {self.labels_dir} "
                f"(expected *{TRAIN_LABEL_SUFFIX}.json or index.json)"
            )
        self.stems = split_stems(
            stems, split, self.cfg.val_ratio, self.cfg.split_seed
        )
        if not self.stems:
            raise RuntimeError(f"split={split!r} is empty (val_ratio={self.cfg.val_ratio})")

        if augment is None:
            augment = split == "train"
        self.augment = bool(augment)
        self.transform = DetSegKPTransform(self.cfg, train=self.augment)

    def __len__(self) -> int:
        return len(self.stems)

    def _load_raw(self, stem: str) -> Dict[str, Any]:
        sample = load_train_sample_from_labels(
            stem,
            self.labels_dir,
            images_dir=self.images_dir,
        )
        image_path = Path(sample["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        image = Image.open(image_path).convert("RGB")
        mask = np.asarray(sample["instance_mask"], dtype=np.uint8)
        if mask.ndim == 3:
            mask = mask[..., 0]
        kps = ensure_kps_xyv(np.asarray(sample["kps"], dtype=np.float32))
        labels = np.asarray(sample["inst_labels"], dtype=np.int64)
        if labels.size == 0:
            labels = np.zeros((kps.shape[0],), dtype=np.int64)
        elif labels.shape[0] != kps.shape[0]:
            n = kps.shape[0]
            out = np.zeros((n,), dtype=np.int64)
            out[: min(n, labels.shape[0])] = labels[: min(n, labels.shape[0])]
            labels = out
        return {"image": image, "mask": mask, "kps": kps, "labels": labels, "stem": stem}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        stem = self.stems[idx]
        raw = self._load_raw(stem)
        image, mask, kps = self.transform(raw["image"], raw["mask"], raw["kps"])
        labels = raw["labels"]

        rgb = np.asarray(image, dtype=np.float32) / 255.0
        h, w = mask.shape
        hs, ws = h // self.stride, w // self.stride
        num_inst = int(kps.shape[0]) if kps.size else 0

        binary_full = instance_mask_to_binary(mask, num_inst)
        boxes = masks_to_boxes_cxcywh(binary_full) if num_inst else np.zeros((0, 4), dtype=np.float32)

        mask_ds = downsample_id_mask(mask, self.stride)
        binary = instance_mask_to_binary(mask_ds, num_inst)

        if num_inst:
            keep = rearrange(binary, "n h w -> n (h w)").sum(axis=1) > 0
            binary = binary[keep]
            boxes = boxes[keep]
            kps = kps[keep]
            labels = labels[keep]
            num_inst = int(keep.sum())

        if num_inst == 0:
            kps = np.zeros((0, 8, 3), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
            boxes = np.zeros((0, 4), dtype=np.float32)
            binary = np.zeros((0, hs, ws), dtype=np.float32)
            kp_maps = np.zeros((0, hs, ws), dtype=np.float32)
        else:
            kps_hm = kps.copy()
            kps_hm[..., 0] /= float(self.stride)
            kps_hm[..., 1] /= float(self.stride)
            kp_maps = kps_to_heatmaps(kps_hm, hs, ws, sigma=self.kp_sigma)

        image_t = rearrange(torch.from_numpy(rgb.copy()), "h w c -> c h w")
        return {
            "image": image_t,
            "masks": torch.from_numpy(np.ascontiguousarray(binary)),
            "kp_maps": torch.from_numpy(np.ascontiguousarray(kp_maps)),
            "labels": torch.from_numpy(np.ascontiguousarray(labels)),
            "boxes": torch.from_numpy(np.ascontiguousarray(boxes)),
            "kps": torch.from_numpy(
                np.ascontiguousarray(rearrange(kps, "n eight three -> n eight three", eight=8, three=3))
            ),
            "stem": stem,
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    images = torch.stack([b["image"] for b in batch])
    return {
        "images": images,
        "targets": [
            {
                "labels": b["labels"],
                "boxes": b["boxes"],
                "masks": b["masks"],
                "kp_maps": b["kp_maps"],
                "kps": b["kps"],
            }
            for b in batch
        ],
        "stems": [b.get("stem", "") for b in batch],
    }


def build_dataloader(
    cfg: DataConfig,
    split: str = "train",
    batch_size: int = 4,
    shuffle: Optional[bool] = None,
    sampler: Optional[torch.utils.data.Sampler] = None,
    dataset: Optional[DetSegKPDataset] = None,
) -> DataLoader:
    if shuffle is None:
        shuffle = split == "train"
    if sampler is not None:
        shuffle = False
    if dataset is None:
        dataset = DetSegKPDataset(cfg, split=split)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
        worker_init_fn=_worker_init_fn if cfg.num_workers > 0 else None,
        drop_last=split == "train",
    )


def build_train_dataloader(
    cfg: Config,
    split: str,
    batch_size: int,
    distributed: bool,
    shuffle: Optional[bool] = None,
) -> DataLoader:
    """Build a train/val dataloader with optional DistributedSampler."""
    if shuffle is None:
        shuffle = split == "train"

    dataset = DetSegKPDataset(cfg.data, split=split)
    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    return build_dataloader(
        cfg.data,
        split=split,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        dataset=dataset,
    )
