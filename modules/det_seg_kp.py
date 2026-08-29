"""
Det + Seg + Keypoint model.

Architecture (SAM3 / MaskFormer inspired):
  1. Vision tower extracts multi-scale image features
  2. FPN pixel decoder fuses them into a shared stride-4 pixel feature map
  3. Object query decoder (Deformable DETR) samples multi-scale FPN via MSDeformAttn
  4. Per-query heads predict: class, bbox, instance mask (dot-product), keypoints
"""
from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import Mlp

from .config import ModelConfig
from .backbone import as_feature_maps, build_vision_tower
from .object_decoder import build_object_query_decoder
from .pixel_decoder import build_pixel_decoder


class DetSegKPHead(nn.Module):
    """Per-query heads for class, mask, and keypoints. Boxes come from the decoder."""

    def __init__(
        self,
        hidden_dim: int,
        num_classes: int,
        pixel_dim: int,
    ):
        super().__init__()
        self.class_head = nn.Linear(hidden_dim, num_classes + 1)  # +1 for no-object
        # Same 2-layer ReLU MLP pattern as decoder bbox_embed (timm.layers.Mlp).
        self.mask_embed = Mlp(hidden_dim, hidden_dim, pixel_dim, act_layer=nn.ReLU)
        self.kp_embed = Mlp(hidden_dim, hidden_dim, pixel_dim, act_layer=nn.ReLU)
        self.pixel_dim = pixel_dim

    def forward(
        self,
        obj_queries: torch.Tensor,
        pixel_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        obj_queries: (B, Q, C)
        pixel_features: (B, pixel_dim, H, W) from pixel decoder
        """
        mask_embed = self.mask_embed(obj_queries)
        kp_embed = self.kp_embed(obj_queries)
        return {
            "pred_logits": self.class_head(obj_queries),
            "pred_masks": torch.einsum("bqd,bdhw->bqhw", mask_embed, pixel_features),
            "pred_kps": torch.einsum("bqd,bdhw->bqhw", kp_embed, pixel_features),
        }


class DetSegKPModel(nn.Module):
    """
    Unified detection + instance segmentation + keypoint model.

    Pipeline:
      image -> vision_tower -> FPN pixel_decoder -> pixel_features (stride 4)
                            -> multi-scale memory (stride >= 8) -> Deformable-DETR decoder
      obj_queries + pixel_features -> det / mask / kp heads
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.vision_tower = build_vision_tower(cfg.vision_tower)
        self.pixel_decoder = build_pixel_decoder(cfg.pixel_decoder)
        self.query_decoder = build_object_query_decoder(cfg.object_query_decoder)
        self.head = DetSegKPHead(
            hidden_dim=cfg.object_query_decoder.hidden_dim,
            num_classes=cfg.num_classes,
            pixel_dim=cfg.pixel_decoder.pixel_dim,
        )

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = as_feature_maps(self.vision_tower(images))
        out_h = images.shape[-2] // self.pixel_decoder.out_stride
        out_w = images.shape[-1] // self.pixel_decoder.out_stride
        pixel_features, fpn = self.pixel_decoder(features, out_size=(out_h, out_w))

        hs, pred_boxes = self.query_decoder(fpn)
        obj_queries = hs[-1]
        outputs = self.head(obj_queries, pixel_features)
        outputs["pred_boxes"] = pred_boxes[-1]
        outputs["aux_outputs"] = [
            {"pred_logits": self.head.class_head(hs[i]), "pred_boxes": pred_boxes[i]}
            for i in range(hs.shape[0] - 1)
        ]
        outputs["pixel_features"] = pixel_features
        outputs["obj_queries"] = obj_queries
        return outputs

    @torch.no_grad()
    def predict(
        self,
        images: torch.Tensor,
        score_threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> List[Dict[str, torch.Tensor]]:
        """Post-process outputs into per-image predictions."""
        self.eval()
        outputs = self.forward(images)
        return postprocess(outputs, score_threshold, mask_threshold)


def postprocess(
    outputs: Dict[str, torch.Tensor],
    score_threshold: float = 0.5,
    mask_threshold: float = 0.5,
) -> List[Dict[str, torch.Tensor]]:
    """Convert raw model outputs to final predictions per image."""
    pred_logits = outputs["pred_logits"]  # (B, Q, C+1)
    pred_boxes = outputs["pred_boxes"]
    pred_masks = outputs["pred_masks"]
    pred_kps = outputs["pred_kps"]

    prob = F.softmax(pred_logits, dim=-1)
    scores, labels = prob[..., :-1].max(dim=-1)

    results = []
    for b in range(pred_logits.shape[0]):
        keep = scores[b] > score_threshold
        results.append({
            "scores": scores[b][keep],
            "labels": labels[b][keep],
            "boxes": pred_boxes[b][keep],
            "masks": (pred_masks[b][keep].sigmoid() > mask_threshold).float(),
            "kps": pred_kps[b][keep].sigmoid(),
        })
    return results
