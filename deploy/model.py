"""Deploy model + CUDA MSDA patch + integral KP decode."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from modules.backbone import as_feature_maps
from modules.config import Config
from modules.det_seg_kp import DetSegKPModel
from modules.utils.box import box_cxcywh_to_xyxy

MSDA_DOMAIN = "det_seg_kp"
IO_NAMES = (
    ("images", "score_threshold", "mask_threshold", "kp_threshold"),
    ("scores", "labels", "boxes", "masks", "keypoints"),
)


def _scalar(t: torch.Tensor) -> torch.Tensor:
    return t if t.ndim == 0 else t.reshape(-1)[0]


# ---- MSDeformAttn (CUDA → ONNX custom op) ---------------------------------


class _MSDAFn(Function):
    @staticmethod
    def forward(ctx, value, spatial_shapes, level_start_index, sampling_loc, attn, im2col_step):
        import modules.ops.ms_deform_atten_2D.ms_deform_attn as msda

        if msda.ms_deform_attn_forward_impl is None:
            raise RuntimeError("ms_deform_atten CUDA extension missing")
        if not value.is_cuda:
            raise RuntimeError("MSDA requires CUDA")
        dtype = value.dtype
        out = msda.ms_deform_attn_forward_impl(
            value.float().contiguous(),
            spatial_shapes.contiguous(),
            level_start_index.contiguous(),
            sampling_loc.float().contiguous(),
            attn.float().contiguous(),
            int(im2col_step),
        )
        return out.to(dtype)

    @staticmethod
    def symbolic(g, value, spatial_shapes, level_start_index, sampling_loc, attn, im2col_step):
        step = int(im2col_step.item()) if isinstance(im2col_step, torch.Tensor) else int(im2col_step)
        return g.op(
            f"{MSDA_DOMAIN}::MSDeformAttn",
            value, spatial_shapes, level_start_index, sampling_loc, attn,
            im2col_step_i=step,
        )


def install_msda_cuda() -> None:
    """Patch training ``ms_deform_attn_core`` to the CUDA deploy op."""
    import modules.ops.ms_deform_atten_2D.ms_deform_attn as msda

    if not msda._MSDA_AVAILABLE or msda.ms_deform_attn_forward_impl is None:
        raise RuntimeError("ms_deform_atten CUDA extension not available")

    def core(value, spatial_shapes, sampling_loc, attn, level_start_index=None, im2col_step=64):
        if level_start_index is None:
            raise RuntimeError("level_start_index required")
        return _MSDAFn.apply(
            value, spatial_shapes, level_start_index, sampling_loc, attn, int(im2col_step)
        )

    msda._MSDA_AVAILABLE = True
    msda.ms_deform_attn_core = core


# ---- Integral keypoints --------------------------------------------------


class IntegralKeypoints(nn.Module):
    """Thresholded soft-argmax; ``xs``/``ys`` cached as buffers."""

    def __init__(self, height: int, width: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.register_buffer(
            "xs", (torch.arange(width, dtype=torch.float32) + 0.5) / width
        )
        self.register_buffer(
            "ys", (torch.arange(height, dtype=torch.float32) + 0.5) / height
        )

    def forward(self, logits: torch.Tensor, thr: torch.Tensor) -> torch.Tensor:
        t = _scalar(thr).to(dtype=logits.dtype)
        w = logits.sigmoid()
        w = w * (w > t).to(w.dtype)
        mass = w.sum((-2, -1)).clamp(min=self.eps)
        xs, ys = self.xs.to(w.dtype), self.ys.to(w.dtype)
        x = (w.sum(-2) * xs).sum(-1) / mass
        y = (w.sum(-1) * ys).sum(-1) / mass
        valid = (mass > self.eps).to(w.dtype)
        return torch.stack([x * valid, y * valid], -1)


# ---- Deploy graph --------------------------------------------------------


class DetSegKPDeploy(nn.Module):
    """backbone → FPN → decoder → heads → in-graph postprocess."""

    def __init__(self, model: DetSegKPModel, heatmap_h: int, heatmap_w: int):
        super().__init__()
        self.vision_tower = model.vision_tower
        self.pixel_decoder = model.pixel_decoder
        self.query_decoder = model.query_decoder
        self.head = model.head
        self.out_stride = int(model.pixel_decoder.out_stride)
        self.kp_decode = IntegralKeypoints(heatmap_h, heatmap_w)

    def forward(self, images, score_thr, mask_thr, kp_thr):
        feats = as_feature_maps(self.vision_tower(images))
        oh, ow = images.shape[-2] // self.out_stride, images.shape[-1] // self.out_stride
        pixel, fpn = self.pixel_decoder(feats, out_size=(oh, ow))
        hs, boxes_all = self.query_decoder(fpn)
        out = self.head(hs[-1], pixel)
        return self._post(
            out["pred_logits"], boxes_all[-1], out["pred_masks"], out["pred_kps"],
            score_thr, mask_thr, kp_thr,
        )

    def _post(self, logits, boxes, masks, kps, score_thr, mask_thr, kp_thr):
        st = _scalar(score_thr).to(logits.dtype)
        mt = _scalar(mask_thr).to(masks.dtype)
        scores, labels = F.softmax(logits, -1)[..., :-1].max(-1)
        keep = (scores > st).to(scores.dtype)
        boxes = box_cxcywh_to_xyxy(boxes).clamp(0, 1) * keep.unsqueeze(-1)
        masks = (masks.sigmoid() > mt).to(masks.dtype) * keep[:, :, None, None]
        kpts = self.kp_decode(kps, kp_thr) * keep.unsqueeze(-1)
        return scores * keep, labels * keep.long(), boxes, masks, kpts


def load_deploy_model(
    checkpoint: str | Path,
    device: str | torch.device = "cuda:0",
    use_ema: bool = True,
    img_height: Optional[int] = None,
    img_width: Optional[int] = None,
) -> Tuple[DetSegKPDeploy, Config]:
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("deploy requires CUDA")
    install_msda_cuda()

    payload: dict[str, Any] = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = payload["cfg"] if isinstance(payload.get("cfg"), Config) else Config()
    cfg.model.vision_tower.pretrained = False

    base = DetSegKPModel(cfg.model)
    state = payload.get("ema") if use_ema and payload.get("ema") is not None else payload.get("model", payload)
    base.load_state_dict(state, strict=False)

    h = int(img_height or cfg.data.img_height)
    w = int(img_width or cfg.data.img_width)
    s = int(cfg.data.stride)
    if h % s or w % s:
        raise ValueError(f"size ({h},{w}) not divisible by stride={s}")

    model = DetSegKPDeploy(base, h // s, w // s).eval().to(device)
    return model, cfg
