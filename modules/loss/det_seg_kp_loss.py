"""Det+Seg+KP multitask loss with Hungarian matching (DETR-style).

Main entry: ``DetSegKPLoss`` / ``build_criterion``.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from scipy.optimize import linear_sum_assignment

from modules.loss.dice_loss import dice_loss
from modules.loss.sigmoid_focal_loss import sigmoid_focal_loss
from modules.utils.box import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """Bipartite matching between queries and ground-truth objects."""

    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
    ):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_boxes: torch.Tensor,
        targets: List[Dict[str, torch.Tensor]],
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        bs, num_queries = pred_logits.shape[:2]
        out_prob = torch.nan_to_num(pred_logits.float().softmax(-1), nan=0.0)
        out_bbox = torch.nan_to_num(pred_boxes.float(), nan=0.5, posinf=1.0, neginf=0.0)

        indices = []
        for b in range(bs):
            tgt_ids = targets[b]["labels"]
            tgt_boxes = targets[b]["boxes"]
            if len(tgt_ids) == 0:
                indices.append((
                    torch.tensor([], dtype=torch.long),
                    torch.tensor([], dtype=torch.long),
                ))
                continue

            cost_class = -out_prob[b][:, tgt_ids]
            cost_bbox = torch.cdist(out_bbox[b].float(), tgt_boxes.float(), p=1)
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox[b].float()),
                box_cxcywh_to_xyxy(tgt_boxes.float()),
            )
            C = (
                self.cost_class * cost_class
                + self.cost_bbox * cost_bbox
                + self.cost_giou * cost_giou
            )
            C = torch.nan_to_num(C, nan=0.0, posinf=1e6, neginf=-1e6).cpu()
            row_ind, col_ind = linear_sum_assignment(C.numpy())
            indices.append((
                torch.as_tensor(row_ind, dtype=torch.long),
                torch.as_tensor(col_ind, dtype=torch.long),
            ))
        return indices


class DetSegKPLoss(nn.Module):
    """Combined loss for detection, segmentation, and keypoints."""

    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: Dict[str, float],
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = 0.1  # down-weight no-object class
        self.register_buffer("empty_weight", empty_weight)

    def _get_src_permutation_idx(
        self, indices: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_idx = torch.cat([
            torch.full_like(src, i) for i, (src, _) in enumerate(indices)
        ])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(
        self, indices: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_idx = torch.cat([
            torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)
        ])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def loss_labels(
        self,
        pred_logits: torch.Tensor,
        targets: List[Dict],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: float,
    ) -> torch.Tensor:
        src_logits = pred_logits
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.long, device=src_logits.device
        )
        idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        target_classes[idx] = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])

        loss = F.cross_entropy(
            rearrange(src_logits, "b q c -> b c q"),
            target_classes,
            self.empty_weight,
            reduction="none",
        )
        return loss.mean()

    def loss_boxes(
        self,
        pred_boxes: torch.Tensor,
        targets: List[Dict],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = self._get_src_permutation_idx(indices)
        src_boxes = pred_boxes[idx]
        target_boxes = torch.cat([
            t["boxes"][J] for t, (_, J) in zip(targets, indices)
        ], dim=0)

        if src_boxes.numel() == 0:
            zero = pred_boxes.sum() * 0
            return zero, zero

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none").sum() / num_boxes
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes),
        )).sum() / num_boxes
        return loss_bbox, loss_giou

    def loss_masks(
        self,
        pred_masks: torch.Tensor,
        targets: List[Dict],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = self._get_src_permutation_idx(indices)
        src_masks = pred_masks[idx]
        target_masks = torch.cat([
            t["masks"][J] for t, (_, J) in zip(targets, indices)
        ], dim=0)

        if src_masks.numel() == 0:
            zero = pred_masks.sum() * 0
            return zero, zero

        # Resize pred masks to target size if needed
        if src_masks.shape[-2:] != target_masks.shape[-2:]:
            src_masks = F.interpolate(
                src_masks.unsqueeze(1),
                size=target_masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        loss_mask = sigmoid_focal_loss(src_masks, target_masks)
        loss_dice = dice_loss(src_masks, target_masks).mean()
        return loss_mask, loss_dice

    def loss_kps(
        self,
        pred_kp: torch.Tensor,
        targets: List[Dict],
        indices: List[Tuple[torch.Tensor, torch.Tensor]],
        num_boxes: float,
    ) -> torch.Tensor:
        idx = self._get_src_permutation_idx(indices)
        src_kp = pred_kp[idx]  # (M, H, W)
        target_kp = torch.cat([
            t["kp_maps"][J] for t, (_, J) in zip(targets, indices)
        ], dim=0)

        if src_kp.numel() == 0:
            return pred_kp.sum() * 0

        if src_kp.shape[-2:] != target_kp.shape[-2:]:
            src_kp = F.interpolate(
                src_kp.unsqueeze(1),
                size=target_kp.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

        return sigmoid_focal_loss(src_kp, target_kp, pos_thresh=0.01)

    def _losses_single(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
        num_boxes: float,
        with_mask_kp: bool,
    ) -> Dict[str, torch.Tensor]:
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        indices = self.matcher(pred_logits, pred_boxes, targets)
        loss_cls = self.loss_labels(pred_logits, targets, indices, num_boxes)
        loss_bbox, loss_giou = self.loss_boxes(pred_boxes, targets, indices, num_boxes)
        losses: Dict[str, torch.Tensor] = {
            "loss_cls": loss_cls,
            "loss_bbox": loss_bbox,
            "loss_giou": loss_giou,
        }
        if with_mask_kp:
            loss_mask, loss_dice = self.loss_masks(
                outputs["pred_masks"], targets, indices, num_boxes
            )
            losses["loss_mask"] = loss_mask
            losses["loss_dice"] = loss_dice
            losses["loss_kp"] = self.loss_kps(
                outputs["pred_kps"], targets, indices, num_boxes
            )
        return losses

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        num_boxes = max(sum(len(t["labels"]) for t in targets), 1)
        losses = self._losses_single(outputs, targets, num_boxes, with_mask_kp=True)
        for i, aux in enumerate(outputs.get("aux_outputs", [])):
            aux_losses = self._losses_single(aux, targets, num_boxes, with_mask_kp=False)
            for key, value in aux_losses.items():
                losses[f"{key}_{i}"] = value

        total = sum(self.weight_dict.get(k, 1.0) * v for k, v in losses.items())
        # Log main-layer terms + one aux aggregate (skip noisy per-layer aux keys).
        main_keys = ("loss_cls", "loss_bbox", "loss_giou", "loss_mask", "loss_dice", "loss_kp")
        loss_dict = {k: losses[k].item() for k in main_keys if k in losses}
        aux_total = sum(
            self.weight_dict.get(k, 1.0) * v
            for k, v in losses.items()
            if k not in main_keys
        )
        loss_dict["loss_aux"] = float(aux_total.item()) if torch.is_tensor(aux_total) else float(aux_total)
        loss_dict["loss_total"] = total.item()
        return total, loss_dict


def build_criterion(cfg) -> DetSegKPLoss:
    matcher = HungarianMatcher(
        cost_class=cfg.train.loss_cls,
        cost_bbox=cfg.train.loss_bbox,
        cost_giou=cfg.train.loss_giou,
    )
    weight_dict = {
        "loss_cls": cfg.train.loss_cls,
        "loss_bbox": cfg.train.loss_bbox,
        "loss_giou": cfg.train.loss_giou,
        "loss_mask": cfg.train.loss_mask,
        "loss_dice": cfg.train.loss_dice,
        "loss_kp": cfg.train.loss_kp,
    }
    for i in range(max(int(cfg.model.num_decoder_layers) - 1, 0)):
        for key in ("loss_cls", "loss_bbox", "loss_giou"):
            weight_dict[f"{key}_{i}"] = weight_dict[key]
    return DetSegKPLoss(
        num_classes=cfg.model.num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
    )
