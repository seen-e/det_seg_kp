#!/usr/bin/env python3
"""ONNX → TensorRT engine via trtexec (+ MSDA plugin)."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _trtexec() -> str:
    for c in (
        os.environ.get("TRTEXEC"),
        shutil.which("trtexec"),
        "/usr/src/tensorrt/bin/trtexec",
        "/usr/local/bin/trtexec",
        "/opt/tensorrt/bin/trtexec",
    ):
        if c and Path(c).is_file():
            return c
    raise SystemExit("trtexec not found (set TRTEXEC=...)")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", required=True)
    p.add_argument("--engine", default="deploy/artifacts/det_seg_kp_fp16.engine")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--workspace", default="4096")
    p.add_argument(
        "--plugins",
        default="deploy/cpp/build/libdet_seg_kp_msda_plugin.so",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--extra", nargs="*", default=[])
    args = p.parse_args()

    onnx = Path(args.onnx)
    if not onnx.is_file():
        raise SystemExit(f"missing onnx: {onnx}")
    engine = Path(args.engine)
    engine.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        _trtexec(),
        f"--onnx={onnx}",
        f"--saveEngine={engine}",
        f"--memPoolSize=workspace:{args.workspace}M",
    ]
    if args.fp16:
        cmd.append("--fp16")
    if args.bf16:
        cmd.append("--bf16")
    if args.plugins:
        cmd.append(f"--plugins={args.plugins}")
    cmd += args.extra

    print(" ".join(cmd))
    if args.dry_run:
        return
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
