"""Train-time GT / Pred visualization at heatmap resolution."""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont

_GT_COLORS = np.array(
    [
        [230, 25, 75],
        [60, 180, 75],
        [67, 99, 216],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [188, 246, 12],
        [250, 190, 190],
        [0, 128, 128],
        [230, 190, 255],
        [154, 99, 36],
        [255, 250, 200],
        [128, 0, 0],
        [170, 255, 195],
    ],
    dtype=np.float32,
)

_TITLE_H = 18
_PANEL_TITLES = ("RGB", "GT-Det", "GT-Seg", "GT-KPS", "Pred-Det", "Pred-Seg", "Pred-KPS")


def _to_numpy_image(image: torch.Tensor) -> np.ndarray:
    """(3, H, W) float [0, 1] -> (H, W, 3) uint8."""
    rgb = image.detach().float().cpu().clamp(0.0, 1.0)
    if rgb.dim() != 3:
        raise ValueError(f"expected image (3, H, W), got {tuple(rgb.shape)}")
    return (rearrange(rgb, "c h w -> h w c").numpy() * 255.0).round().astype(np.uint8)


def _resize_rgb(rgb: np.ndarray, height: int, width: int) -> np.ndarray:
    img = Image.fromarray(rgb, mode="RGB")
    return np.asarray(img.resize((width, height), Image.BILINEAR), dtype=np.uint8)


def _gt_colors(n: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)
    reps = int(np.ceil(n / len(_GT_COLORS)))
    return np.tile(_GT_COLORS, (reps, 1))[:n] / 255.0


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


def render_gt_pred_strip(
    image: torch.Tensor,
    target: Dict[str, torch.Tensor],
    pred: Dict[str, torch.Tensor],
    stride: int,
    score_threshold: float = 0.5,
) -> np.ndarray:
    """One sample: RGB | GT-Det | GT-Seg | GT-KPS | Pred-Det | Pred-Seg | Pred-KPS.

    Uses last-layer predictions. Matched Pred instances share GT colors.
    Pred-Det draws matched boxes with score above ``score_threshold``.
    GT-KPS / Pred-KPS are colorized heatmaps on black.
    """
    rgb_full = _to_numpy_image(image)
    masks_t = target.get("masks")
    kp_maps_t = target.get("kp_maps")
    pred_masks_t = pred["pred_masks"]
    pred_kps_t = pred["pred_kps"]

    if masks_t is not None and masks_t.numel() > 0:
        hs, ws = int(masks_t.shape[-2]), int(masks_t.shape[-1])
    elif pred_masks_t is not None and pred_masks_t.numel() > 0:
        hs, ws = int(pred_masks_t.shape[-2]), int(pred_masks_t.shape[-1])
    else:
        hs = max(int(rgb_full.shape[0]) // int(stride), 1)
        ws = max(int(rgb_full.shape[1]) // int(stride), 1)

    rgb = _resize_rgb(rgb_full, hs, ws)

    gt_masks = (
        _as_hw_masks(masks_t, hs, ws)
        if masks_t is not None
        else np.zeros((0, hs, ws), dtype=np.float32)
    )
    gt_heat = (
        _as_hw_masks(kp_maps_t, hs, ws)
        if kp_maps_t is not None
        else np.zeros((0, hs, ws), dtype=np.float32)
    )
    boxes_t = target.get("boxes")
    if boxes_t is not None and boxes_t.numel() > 0:
        gt_boxes = boxes_t.detach().float().cpu().numpy()
    else:
        gt_boxes = np.zeros((0, 4), dtype=np.float32)

    n_gt = max(gt_masks.shape[0], gt_boxes.shape[0], gt_heat.shape[0])
    gt_col = _gt_colors(n_gt)

    pred_logits = pred["pred_logits"]
    pred_boxes_t = pred["pred_boxes"]
    pred_boxes = pred_boxes_t.detach().float().cpu().numpy()
    scores = _object_scores(pred_logits)
    src_idx, tgt_idx = _match_queries_to_gt(pred_logits, pred_boxes_t, target)

    # Pred-Det: matched boxes, GT colors, score filter
    vis_boxes = np.zeros((0, 4), dtype=np.float32)
    vis_col = np.zeros((0, 3), dtype=np.float32)
    if src_idx.size:
        keep = scores[src_idx] > score_threshold
        src_keep = src_idx[keep]
        tgt_keep = tgt_idx[keep]
        if src_keep.size:
            vis_boxes = pred_boxes[src_keep]
            vis_col = gt_col[tgt_keep]

    pred_mask_np = torch.sigmoid(pred_masks_t.detach().float().cpu())
    pred_kp_np = torch.sigmoid(pred_kps_t.detach().float().cpu())
    pred_masks = _as_hw_masks(pred_mask_np, hs, ws)
    pred_heats = _as_hw_masks(pred_kp_np, hs, ws)
    pred_masks_inst = _aligned_to_gt_slots(pred_masks, src_idx, tgt_idx, n_gt)
    pred_heats_inst = _aligned_to_gt_slots(pred_heats, src_idx, tgt_idx, n_gt)

    gt_det = _overlay_boxes(rgb, gt_boxes, gt_col)
    gt_seg = _overlay_masks(rgb, gt_masks, gt_col, alpha=0.55)
    gt_kps_img = _render_heatmaps(gt_heat, gt_col, hs, ws)

    pred_det = _overlay_boxes(rgb, vis_boxes, vis_col)
    pred_seg = _overlay_masks(rgb, pred_masks_inst, gt_col, alpha=0.55)
    pred_kps_img = _render_heatmaps(pred_heats_inst, gt_col, hs, ws)

    panels = [rgb, gt_det, gt_seg, gt_kps_img, pred_det, pred_seg, pred_kps_img]
    titled = [_panel_with_title(p, name) for p, name in zip(panels, _PANEL_TITLES)]
    return np.concatenate(titled, axis=1)


def save_gt_pred_strip(
    strip: np.ndarray,
    output_dir: str,
    epoch: int,
    step: int,
) -> str:
    """Write strip PNG to ``{output_dir}/vis/epoch####_step######.png``."""
    vis_dir = os.path.join(output_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)
    path = os.path.join(vis_dir, f"epoch{epoch:04d}_step{step:06d}.png")
    Image.fromarray(strip, mode="RGB").save(path)
    return path


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
    pred = {
        "pred_logits": outputs["pred_logits"][0],
        "pred_boxes": outputs["pred_boxes"][0],
        "pred_masks": outputs["pred_masks"][0],
        "pred_kps": outputs["pred_kps"][0],
    }
    strip = render_gt_pred_strip(
        image, target, pred, stride=stride, score_threshold=score_threshold,
    )
    path = save_gt_pred_strip(strip, output_dir, epoch, step)
    if wandb_run is not None:
        wandb_run.log(
            {"vis/gt_pred": wandb_run.Image(strip, caption=os.path.basename(path))},
            step=step,
        )
    return path
