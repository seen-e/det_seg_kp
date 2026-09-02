#!/usr/bin/env python3
"""Build ONNX from DetSegKPDeploy (CUDA MSDA custom op)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deploy.model import IO_NAMES, MSDA_DOMAIN, load_deploy_model  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", default="deploy/artifacts/det_seg_kp.onnx")
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--score-threshold", type=float, default=0.5)
    p.add_argument("--mask-threshold", type=float, default=0.5)
    p.add_argument("--kp-threshold", type=float, default=0.1)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("need --device cuda:N")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    b, h, w = args.batch_size, args.height, args.width
    model, cfg = load_deploy_model(
        args.checkpoint, device, use_ema=not args.no_ema, img_height=h, img_width=w
    )

    inputs = (
        torch.zeros(b, 3, h, w, device=device),
        torch.tensor([args.score_threshold], device=device),
        torch.tensor([args.mask_threshold], device=device),
        torch.tensor([args.kp_threshold], device=device),
    )
    if args.verify:
        with torch.no_grad():
            for n, t in zip(IO_NAMES[1], model(*inputs)):
                print(f"  {n}: {tuple(t.shape)}")

    with torch.no_grad():
        torch.onnx.export(
            model, inputs, str(out),
            input_names=list(IO_NAMES[0]),
            output_names=list(IO_NAMES[1]),
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
            custom_opsets={MSDA_DOMAIN: 1},
        )
    stride = cfg.data.stride
    print(f"wrote {out}  mask={h // stride}x{w // stride}  node={MSDA_DOMAIN}::MSDeformAttn")


if __name__ == "__main__":
    main()
