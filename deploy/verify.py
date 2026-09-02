#!/usr/bin/env python3
"""CUDA smoke test for DetSegKPDeploy."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deploy.model import DetSegKPDeploy, install_msda_cuda, load_deploy_model  # noqa: E402
from modules.config import Config  # noqa: E402
from modules.det_seg_kp import DetSegKPModel  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="")
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--score-threshold", type=float, default=0.5)
    p.add_argument("--mask-threshold", type=float, default=0.5)
    p.add_argument("--kp-threshold", type=float, default=0.1)
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("need --device cuda:N")
    h, w = args.height, args.width

    if args.checkpoint:
        model, _ = load_deploy_model(args.checkpoint, device, img_height=h, img_width=w)
    else:
        install_msda_cuda()
        cfg = Config()
        cfg.model.vision_tower.pretrained = False
        s = cfg.data.stride
        model = DetSegKPDeploy(DetSegKPModel(cfg.model).eval(), h // s, w // s).to(device).eval()

    x = (
        torch.rand(1, 3, h, w, device=device),
        torch.tensor([args.score_threshold], device=device),
        torch.tensor([args.mask_threshold], device=device),
        torch.tensor([args.kp_threshold], device=device),
    )
    with torch.no_grad():
        scores, _, boxes, masks, kps = model(*x)
    print(f"valid={(scores[0] > 0).sum().item()}  boxes={tuple(boxes.shape)}  "
          f"masks={tuple(masks.shape)}  kps={tuple(kps.shape)}")


if __name__ == "__main__":
    main()
