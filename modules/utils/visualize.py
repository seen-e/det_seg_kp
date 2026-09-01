"""Train-time GT / Pred visualization at heatmap resolution."""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont

_TITLE_H = 18
_PANEL_TITLES = ("RGB", "GT-Det", "GT-Seg", "GT-KPS", "Pred-Det", "Pred-Seg", "Pred-KPS")
# Golden-ratio conjugate: successive hues stay far apart on the circle.
_GOLDEN_RATIO_CONJUGATE = 0.618033988749895


def _hsv_to_rgb(h: float, s: float, v: float) -> np.ndarray:
    """HSV in [0, 1] -> RGB float32 in [0, 1]."""
    i = int(h * 6.0) % 6
    f = h * 6.0 - int(h * 6.0)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    if i == 0:
        r, g, b = v, t, p
    elif i == 1:
        r, g, b = q, v, p
    elif i == 2:
        r, g, b = p, v, t
    elif i == 3:
        r, g, b = p, q, v
    elif i == 4:
        r, g, b = t, p, v
    else:
        r, g, b = v, p, q
    return np.array([r, g, b], dtype=np.float32)


def _id_to_color(idx: int, *, sat: float = 0.72, val: float = 0.95) -> np.ndarray:
    """Map instance id -> distinct RGB via golden-ratio hue walk."""
    hue = (float(idx) * _GOLDEN_RATIO_CONJUGATE) % 1.0
    return _hsv_to_rgb(hue, sat, val)


def _gt_colors(n: int) -> np.ndarray:
    """(n, 3) float RGB in [0, 1]; unlimited distinct-ish colors by id."""
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack([_id_to_color(i) for i in range(n)], axis=0)


def _to_numpy_image(image: torch.Tensor) -> np.ndarray:
    """(3, H, W) float [0, 1] -> (H, W, 3) uint8."""
    rgb = image.detach().float().cpu().clamp(0.0, 1.0)
    if rgb.dim() != 3:
        raise ValueError(f"expected image (3, H, W), got {tuple(rgb.shape)}")
    return (rearrange(rgb, "c h w -> h w c").numpy() * 255.0).round().astype(np.uint8)


def _resize_rgb(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    img = Image.fromarray(rgb, mode="RGB")
    return np.asarray(img.resize((width, height), Image.BILINEAR), dtype=np.uint8)


def _as_hw_masks(t: torch.Tensor, height: int, width: int) -> np.ndarray:
    """(N, H, W) or empty -> float32 (N, height, width) in [0, 1]."""
    if t is None or t.numel() == 0:
        return np.zeros((0, height, width), dtype=np.float32)
    x = t.detach().float().cpu()
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if x.shape[-2:] != (height, width):
        x = F.interpolate(
            x.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False
        ).squeeze(1)
    return x.clamp(0.0, 1.0).numpy()


def _overlay_masks(
    rgb: np.ndarray,
    masks: np.ndarray,
    colors: np.ndarray,
    weights: Optional[np.ndarray] = None,
    alpha: float = 0.5,
) -> np.ndarray:
    out = rgb.astype(np.float32)
    n = int(masks.shape[0]) if masks.ndim == 3 else 0
    if n == 0:
        return rgb.copy()
    if weights is None:
        weights = np.ones((n,), dtype=np.float32)
    for i in range(n):
        m = np.clip(masks[i] * float(weights[i]), 0.0, 1.0)[..., None]
        a = alpha * m
        color = colors[i % len(colors)] * 255.0
        out = out * (1.0 - a) + color * a
    return np.clip(out, 0.0, 255.0).round().astype(np.uint8)


def _render_heatmaps(
    maps: np.ndarray,
    colors: np.ndarray,
    height: int,
    width: int,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Colorize heatmaps on a black canvas (no RGB overlay)."""
    out = np.zeros((height, width, 3), dtype=np.float32)
    n = int(maps.shape[0]) if maps.ndim == 3 else 0
    if n == 0:
        return out.round().astype(np.uint8)
    if weights is None:
        weights = np.ones((n,), dtype=np.float32)
    for i in range(n):
        m = np.clip(maps[i] * float(weights[i]), 0.0, 1.0)[..., None]
        color = colors[i % len(colors)] * 255.0
        np.maximum(out, color * m, out=out)
    return np.clip(out, 0.0, 255.0).round().astype(np.uint8)


def _overlay_boxes(
    rgb: np.ndarray,
    boxes_cxcywh: np.ndarray,
    colors: np.ndarray,
) -> np.ndarray:
    img = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(img)
    h, w = rgb.shape[:2]
    n = int(boxes_cxcywh.shape[0]) if boxes_cxcywh.size else 0
    if n == 0:
        return rgb.copy()
    for i in range(n):
        cx, cy, bw, bh = [float(v) for v in boxes_cxcywh[i]]
        x0 = (cx - 0.5 * bw) * w
        y0 = (cy - 0.5 * bh) * h
        x1 = (cx + 0.5 * bw) * w
        y1 = (cy + 0.5 * bh) * h
        color = tuple(int(c * 255) for c in colors[i % len(colors)])
        draw.rectangle([x0, y0, x1, y1], outline=color, width=1)
    return np.asarray(img, dtype=np.uint8)


def _panel_with_title(rgb: np.ndarray, title: str) -> np.ndarray:
    h, w = rgb.shape[:2]
    canvas = np.zeros((h + _TITLE_H, w, 3), dtype=np.uint8)
    canvas[:_TITLE_H] = 24
    canvas[_TITLE_H:] = rgb
    img = Image.fromarray(canvas, mode="RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.text((4, 3), title, fill=(240, 240, 240), font=font)
    return np.asarray(img, dtype=np.uint8)


def _object_scores(pred_logits: torch.Tensor) -> np.ndarray:
    """Per-query objectness = max class prob excluding no-object. (Q,)"""
    prob = F.softmax(pred_logits.detach().float().cpu(), dim=-1)
    scores = prob[..., :-1].max(dim=-1).values
    return scores.numpy()


_VIS_MATCHER = None


def _get_vis_matcher():
    """Reuse one HungarianMatcher across visualization calls."""
    global _VIS_MATCHER
    if _VIS_MATCHER is None:
        from modules.loss.det_seg_kp_loss import HungarianMatcher

        _VIS_MATCHER = HungarianMatcher()
    return _VIS_MATCHER


def _match_queries_to_gt(
    pred_logits: torch.Tensor,
    pred_boxes: torch.Tensor,
    target: Dict[str, torch.Tensor],
) -> Tuple[np.ndarray, np.ndarray]:
    """Hungarian match last-layer queries to GT. Returns (src_idx, tgt_idx) as int arrays."""
    indices = _get_vis_matcher()(
        pred_logits.unsqueeze(0),
        pred_boxes.unsqueeze(0),
        [target],
    )
    src_idx, tgt_idx = indices[0]
    if src_idx.numel() == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    return src_idx.cpu().numpy().astype(np.int64), tgt_idx.cpu().numpy().astype(np.int64)


def _aligned_to_gt_slots(
    pred_maps: np.ndarray,
    src_idx: np.ndarray,
    tgt_idx: np.ndarray,
    n_gt: int,
) -> np.ndarray:
    """Place matched query maps into GT instance slots (same color layout as GT)."""
    if n_gt <= 0 or pred_maps.shape[0] == 0 or src_idx.size == 0:
        hs = pred_maps.shape[-2] if pred_maps.ndim == 3 else 1
        ws = pred_maps.shape[-1] if pred_maps.ndim == 3 else 1
        return np.zeros((max(n_gt, 0), hs, ws), dtype=np.float32)
    matched = np.zeros((n_gt, *pred_maps.shape[-2:]), dtype=np.float32)
    for q, g in zip(src_idx, tgt_idx):
        matched[int(g)] = pred_maps[int(q)]
    return matched


def _as_boxes_np(t: Optional[torch.Tensor]) -> np.ndarray:
    """(N, 4) cxcywh or empty (0, 4)."""
    if t is None or t.numel() == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return t.detach().float().cpu().numpy()


def _heatmap_hw(
    masks_t: Optional[torch.Tensor],
    pred_masks_t: torch.Tensor,
    image_hw: Tuple[int, int],
    stride: int,
) -> Tuple[int, int]:
    """Prefer GT / pred mask spatial size; else image // stride."""
    for t in (masks_t, pred_masks_t):
        if t is not None and t.numel() > 0:
            return int(t.shape[-2]), int(t.shape[-1])
    h, w = image_hw
    return max(h // int(stride), 1), max(w // int(stride), 1)


def render_gt_pred_strip(
    image: torch.Tensor,
    target: Dict[str, torch.Tensor],
    pred: Dict[str, torch.Tensor],
    stride: int,
    score_threshold: float = 0.5,
    *,
    match_to_gt: bool = True,
) -> np.ndarray:
    """One sample: RGB | GT-Det | GT-Seg | GT-KPS | Pred-Det | Pred-Seg | Pred-KPS.

    ``match_to_gt`` (train): Pred aligned to GT slots/colors, matched boxes only.
    Otherwise (val): all queries with score > ``score_threshold`` (FPs included).
    """
    rgb_full = _to_numpy_image(image)
    masks_t, kp_maps_t = target.get("masks"), target.get("kp_maps")
    pred_masks_t, pred_kps_t = pred["pred_masks"], pred["pred_kps"]
    hs, ws = _heatmap_hw(masks_t, pred_masks_t, rgb_full.shape[:2], stride)
    rgb = _resize_rgb(rgb_full, hs, ws)

    gt_masks = _as_hw_masks(masks_t, hs, ws)
    gt_heat = _as_hw_masks(kp_maps_t, hs, ws)
    gt_boxes = _as_boxes_np(target.get("boxes"))
    n_gt = max(gt_masks.shape[0], gt_boxes.shape[0], gt_heat.shape[0])
    gt_col = _gt_colors(n_gt)

    pred_logits, pred_boxes_t = pred["pred_logits"], pred["pred_boxes"]
    pred_boxes = _as_boxes_np(pred_boxes_t)
    scores = _object_scores(pred_logits)
    pred_masks = _as_hw_masks(pred_masks_t.detach().float().cpu().sigmoid(), hs, ws)
    pred_heats = _as_hw_masks(pred_kps_t.detach().float().cpu().sigmoid(), hs, ws)

    if match_to_gt:
        src_idx, tgt_idx = _match_queries_to_gt(pred_logits, pred_boxes_t, target)
        keep = scores[src_idx] > score_threshold if src_idx.size else np.zeros(0, dtype=bool)
        q_idx, t_idx = src_idx[keep], tgt_idx[keep]
        pred_col = gt_col[t_idx]
        pred_masks_vis = _aligned_to_gt_slots(pred_masks, src_idx, tgt_idx, n_gt)
        pred_heats_vis = _aligned_to_gt_slots(pred_heats, src_idx, tgt_idx, n_gt)
        map_col = gt_col
    else:
        q_idx = np.flatnonzero(scores > score_threshold)
        q_idx = q_idx[np.argsort(-scores[q_idx])]
        pred_col = _gt_colors(len(q_idx))
        pred_masks_vis, pred_heats_vis = pred_masks[q_idx], pred_heats[q_idx]
        map_col = pred_col

    panels = [
        rgb,
        _overlay_boxes(rgb, gt_boxes, gt_col),
        _overlay_masks(rgb, gt_masks, gt_col, alpha=0.55),
        _render_heatmaps(gt_heat, gt_col, hs, ws),
        _overlay_boxes(rgb, pred_boxes[q_idx], pred_col),
        _overlay_masks(rgb, pred_masks_vis, map_col, alpha=0.55),
        _render_heatmaps(pred_heats_vis, map_col, hs, ws),
    ]
    return np.concatenate(
        [_panel_with_title(p, name) for p, name in zip(panels, _PANEL_TITLES)],
        axis=1,
    )


_JPEG_QUALITY = 92


def _save_strip_jpeg(strip: np.ndarray, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(strip, mode="RGB").save(path, format="JPEG", quality=_JPEG_QUALITY)
    return path


def save_gt_pred_strip(
    strip: np.ndarray,
    output_dir: str,
    epoch: int,
    step: int,
) -> str:
    """Write strip JPEG to ``{output_dir}/vis/epoch####_step######.jpg``."""
    vis_dir = os.path.join(output_dir, "vis")
    path = os.path.join(vis_dir, f"epoch{epoch:04d}_step{step:06d}.jpg")
    return _save_strip_jpeg(strip, path)


def save_val_gt_pred_strip(
    strip: np.ndarray,
    output_dir: str,
    epoch: int,
    sample_name: str,
) -> str:
    """Write val strip JPEG to ``{output_dir}/vis_val/epoch_{epoch:05d}/{sample_name}.jpg``."""
    vis_dir = os.path.join(output_dir, "vis_val", f"epoch_{epoch:05d}")
    safe = str(sample_name).replace("/", "_").replace("\\", "_").strip() or "sample"
    if not safe.lower().endswith((".jpg", ".jpeg", ".png")):
        safe = f"{safe}.jpg"
    path = os.path.join(vis_dir, safe)
    return _save_strip_jpeg(strip, path)


def _strip_from_outputs(
    image: torch.Tensor,
    target: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    *,
    stride: int,
    score_threshold: float,
    sample_index: int = 0,
    match_to_gt: bool = True,
) -> np.ndarray:
    pred = {
        "pred_logits": outputs["pred_logits"][sample_index],
        "pred_boxes": outputs["pred_boxes"][sample_index],
        "pred_masks": outputs["pred_masks"][sample_index],
        "pred_kps": outputs["pred_kps"][sample_index],
    }
    return render_gt_pred_strip(
        image,
        target,
        pred,
        stride=stride,
        score_threshold=score_threshold,
        match_to_gt=match_to_gt,
    )


@torch.no_grad()
def log_train_visualization(
    image: torch.Tensor,
    target: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    *,
    stride: int,
    output_dir: str,
    epoch: int,
    step: int,
    wandb_run: Optional[object] = None,
    score_threshold: float = 0.5,
) -> str:
    """Render and save one-sample GT/Pred strip from last-layer outputs."""
    strip = _strip_from_outputs(
        image,
        target,
        outputs,
        stride=stride,
        score_threshold=score_threshold,
        match_to_gt=True,
    )
    path = save_gt_pred_strip(strip, output_dir, epoch, step)
    if wandb_run is not None:
        wandb_run.log(
            {"vis/gt_pred": wandb_run.Image(strip, caption=os.path.basename(path))},
            step=step,
        )
    return path


@torch.no_grad()
def log_val_visualization(
    image: torch.Tensor,
    target: Dict[str, torch.Tensor],
    outputs: Dict[str, torch.Tensor],
    *,
    stride: int,
    output_dir: str,
    epoch: int,
    sample_name: str,
    sample_index: int = 0,
    wandb_run: Optional[object] = None,
    global_step: Optional[int] = None,
    score_threshold: float = 0.5,
) -> str:
    """Render and save one val-sample strip; Pred filtered by score only (FP visible).

    Writes to ``{output_dir}/vis_val/epoch_{epoch:05d}/{sample_name}.jpg``.
    """
    strip = _strip_from_outputs(
        image,
        target,
        outputs,
        stride=stride,
        score_threshold=score_threshold,
        sample_index=sample_index,
        match_to_gt=False,
    )
    path = save_val_gt_pred_strip(strip, output_dir, epoch, sample_name)
    if wandb_run is not None:
        wandb_run.log(
            {"vis/val_gt_pred": wandb_run.Image(strip, caption=os.path.basename(path))},
            step=global_step if global_step is not None else epoch,
        )
    return path
