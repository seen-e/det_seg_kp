"""Dataset and dataloader for det + seg + keypoint.

On-disk layout from ``gen_and_save_from_tree``::

    {data_root}/index.json
    {data_root}/{rel_dir}/{stem}.jpg
    {data_root}/{rel_dir}/{stem}_instance_mask.png
    {data_root}/{rel_dir}/{stem}_train.json

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
    ensure_kps_xyv,
    kp_corner_valid,
    kp_xy,
    load_train_sample_from_labels,
)
from modules.utils import gaussian_heatmap


def _sample_id(rel_dir: str, stem: str) -> str:
    rel_dir = (rel_dir or "").strip().strip("/\\")
    return f"{rel_dir}/{stem}" if rel_dir else stem


def _entries_from_index(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [e for e in payload if isinstance(e, dict) and "stem" in e]
    if isinstance(payload, dict):
        samples = payload.get("samples", [])
        if isinstance(samples, list):
            return [e for e in samples if isinstance(e, dict) and "stem" in e]
    return []


def instance_mask_to_binary(mask: np.ndarray, num_instances: int) -> np.ndarray:
    """
    Convert an id map (H, W) with labels 0..k-1 / bg=255 into (k, H, W) float32
    binary masks used by the mask / dice losses.
    """
    h, w = mask.shape
    binary = np.zeros((num_instances, h, w), dtype=np.float32)
    for i in range(num_instances):
        binary[i] = (mask == i).astype(np.float32)
    return binary


def kps_to_heatmap(
    kps: np.ndarray,
    height: int,
    width: int,
    sigma: float = 2.0,
    threshold: Optional[float] = 0.01,
    heatmap: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Draw up to 8 corners onto a single heatmap.

    Args:
        kps: (8, 3) or (8, 2) legacy; a point is skipped unless v == 0.
             Coordinates are in this heatmap's pixel space.
    Returns:
        heatmap: (H, W) float32 in [0, 1]
    """
    heatmap = np.zeros((height, width), dtype=np.float32) if heatmap is None else heatmap
    kps = np.asarray(kps, dtype=np.float32)
    if kps.ndim == 1:
        kps = rearrange(kps, "c -> 1 c")
    for kp in kps:
        if not kp_corner_valid(kp):
            continue
        x, y = kp_xy(kp)
        blob = gaussian_heatmap((x, y), (height, width), sigma=sigma, threshold=threshold)
        np.maximum(heatmap, blob, out=heatmap)
    return heatmap


def kps_to_heatmaps(
    kps: np.ndarray,
    height: int,
    width: int,
    sigma: float = 2.0,
    threshold: Optional[float] = 0.01,
) -> np.ndarray:
    """(k, 8, 3) -> (k, H, W); one fused heatmap per instance."""
    kps = np.asarray(kps, dtype=np.float32)
    out = np.zeros((kps.shape[0], height, width), dtype=np.float32)
    for i in range(kps.shape[0]):
        kps_to_heatmap(
            kps[i], height, width, sigma=sigma, threshold=threshold, heatmap=out[i]
        )
    return out


def boxes_xyxy_to_cxcywh_norm(boxes: np.ndarray, height: int, width: int) -> np.ndarray:
    """Exclusive xyxy (N,4) → normalized cxcywh in [0, 1]. Empty → small centered box."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    n = boxes.shape[0]
    out = np.zeros((n, 4), dtype=np.float32)
    if n == 0:
        return out
    w = float(width)
    h = float(height)
    bw = boxes[:, 2] - boxes[:, 0]
    bh = boxes[:, 3] - boxes[:, 1]
    valid = (bw > 0) & (bh > 0)
    out[:, :] = (0.5, 0.5, 0.1, 0.1)
    out[valid, 0] = 0.5 * (boxes[valid, 0] + boxes[valid, 2]) / w
    out[valid, 1] = 0.5 * (boxes[valid, 1] + boxes[valid, 3]) / h
    out[valid, 2] = bw[valid] / w
    out[valid, 3] = bh[valid] / h
    return out


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


def discover_samples(data_root: Path) -> List[Dict[str, Any]]:
    """Load samples from ``{data_root}/index.json`` (``stem`` + ``rel_dir``).

    Each entry: ``id``, ``stem``, ``sample_dir`` (dir with jpg / mask / ``*_train.json``).
    """
    data_root = Path(data_root)
    index_path = data_root / "index.json"
    if not index_path.is_file():
        return []
    with index_path.open(encoding="utf-8") as f:
        entries = _entries_from_index(json.load(f))
    out: List[Dict[str, Any]] = []
    for e in entries:
        stem = str(e["stem"])
        rel_dir = str(e.get("rel_dir") or "")
        sample_dir = data_root / rel_dir if rel_dir else data_root
        out.append(
            {
                "id": _sample_id(rel_dir, stem),
                "stem": stem,
                "sample_dir": sample_dir,
            }
        )
    return out


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


def split_samples(
    samples: Sequence[Dict[str, Any]],
    split: str,
    val_ratio: float,
    seed: int,
) -> List[Dict[str, Any]]:
    """Train/val split over sample records using stable ``id`` keys."""
    ids = [s["id"] for s in samples]
    keep = set(split_stems(ids, split, val_ratio, seed))
    return [s for s in samples if s["id"] in keep]


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
        self.kp_threshold = self.cfg.kp_threshold
        if self.img_w % self.stride != 0 or self.img_h % self.stride != 0:
            raise ValueError(
                f"img_width={self.img_w}, img_height={self.img_h} must be "
                f"divisible by stride={self.stride}"
            )
        self.out_w = self.img_w // self.stride
        self.out_h = self.img_h // self.stride

        self.data_root = Path(self.cfg.data_root)
        samples = discover_samples(self.data_root)
        if not samples:
            raise FileNotFoundError(
                f"No training samples under {self.data_root} "
                f"(expected {self.data_root / 'index.json'} with samples[].stem/rel_dir)"
            )
        self.samples = split_samples(
            samples, split, self.cfg.val_ratio, self.cfg.split_seed
        )
        if not self.samples:
            raise RuntimeError(f"split={split!r} is empty (val_ratio={self.cfg.val_ratio})")
        self.stems = [s["id"] for s in self.samples]

        if augment is None:
            augment = split == "train"
        self.augment = bool(augment)
        self.transform = DetSegKPTransform(self.cfg, train=self.augment)

    def __len__(self) -> int:
        return len(self.samples)

    def _load_raw(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        sample_dir = Path(entry["sample_dir"])
        sample = load_train_sample_from_labels(
            entry["stem"],
            sample_dir,
            images_dir=sample_dir,
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
        boxes = sample.get("boxes")
        if boxes is None:
            boxes = np.zeros((kps.shape[0], 4), dtype=np.float32)
        else:
            boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
            if boxes.shape[0] != kps.shape[0]:
                boxes = np.zeros((kps.shape[0], 4), dtype=np.float32)
        return {
            "image": image,
            "mask": mask,
            "kps": kps,
            "labels": labels,
            "boxes": boxes,
            "stem": entry["id"],
        }

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        entry = self.samples[idx]
        raw = self._load_raw(entry)
        image, mask, kps, labels, boxes_xyxy = self.transform(
            raw["image"], raw["mask"], raw["kps"], raw["labels"], raw["boxes"],
        )

        rgb = np.asarray(image, dtype=np.float32) / 255.0
        h, w = mask.shape
        hs, ws = h // self.stride, w // self.stride
        num_inst = int(kps.shape[0]) if kps.size else 0

        boxes = boxes_xyxy_to_cxcywh_norm(boxes_xyxy, h, w)

        mask_ds = downsample_id_mask(mask, self.stride)
        binary = instance_mask_to_binary(mask_ds, num_inst)

        # Drop instances that vanished after nearest downsample (filters already
        # ran at full res; this is a cheap heatmap-empty safety net).
        keep = rearrange(binary, "n h w -> n (h w)").sum(axis=1) > 0
        binary = binary[keep]
        boxes = boxes[keep]
        kps = kps[keep]
        labels = labels[keep]

        kps_hm = kps.copy()
        kps_hm[..., 0] /= float(self.stride)
        kps_hm[..., 1] /= float(self.stride)
        kp_maps = kps_to_heatmaps(
            kps_hm, hs, ws, sigma=self.kp_sigma, threshold=self.kp_threshold
        )

        image_t = rearrange(torch.from_numpy(rgb.copy()), "h w c -> c h w")
        return {
            "image": image_t,
            "masks": torch.from_numpy(np.ascontiguousarray(binary)),
            "kp_maps": torch.from_numpy(np.ascontiguousarray(kp_maps)),
            "labels": torch.from_numpy(np.ascontiguousarray(labels)),
            "boxes": torch.from_numpy(np.ascontiguousarray(boxes)),
            "kps": torch.from_numpy(np.ascontiguousarray(kps)),
            "stem": entry["id"],
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack images; keep per-sample targets (variable N) under ``targets``."""
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
    cfg: Config,
    split: str,
    batch_size: int,
    distributed: bool = False,
    shuffle: Optional[bool] = None,
) -> DataLoader:
    """Build train or val ``DataLoader`` (``split`` is ``"train"`` or ``"val"``).

    Defaults: train shuffles and drops last batch; val does neither.
    Uses ``DistributedSampler`` when ``distributed`` is True.
    """
    if shuffle is None:
        shuffle = split == "train"

    dataset = DetSegKPDataset(cfg.data, split=split)
    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    if sampler is not None:
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=cfg.data.num_workers > 0,
        worker_init_fn=_worker_init_fn if cfg.data.num_workers > 0 else None,
        drop_last=split == "train",
    )
