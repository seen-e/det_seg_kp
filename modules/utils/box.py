"""Box format conversion and GIoU."""
from __future__ import annotations

import torch


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (cx, cy, w, h) to (x0, y0, x1, y1); sanitizes NaN/Inf."""
    boxes = torch.nan_to_num(boxes, nan=0.5, posinf=1.0, neginf=0.0)
    cx, cy, w, h = boxes.unbind(-1)
    w = w.clamp(min=0.0)
    h = h.clamp(min=0.0)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Convert (x0, y0, x1, y1) to (cx, cy, w, h)."""
    x0, y0, x1, y1 = boxes.unbind(-1)
    return torch.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], dim=-1)


def _sanitize_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """Ensure finite xyxy with x1>=x0, y1>=y0 (early-train / AMP safe)."""
    boxes = torch.nan_to_num(boxes, nan=0.0, posinf=1.0, neginf=0.0)
    x0y0, x1y1 = boxes[:, :2], boxes[:, 2:]
    lt = torch.min(x0y0, x1y1)
    rb = torch.max(x0y0, x1y1)
    return torch.cat([lt, rb], dim=-1)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute GIoU between two sets of boxes in xyxy format."""
    boxes1 = _sanitize_xyxy(boxes1)
    boxes2 = _sanitize_xyxy(boxes2)

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)

    lt_c = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb_c = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[:, :, 0] * wh_c[:, :, 1]
    giou = iou - (area_c - union) / (area_c + 1e-6)
    return torch.nan_to_num(giou, nan=-1.0, posinf=0.0, neginf=-1.0)
