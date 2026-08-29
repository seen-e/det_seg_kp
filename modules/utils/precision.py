"""Configurable training precision: fp32 / fp16 / bf16 / amp."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.amp import GradScaler, autocast

PRECISION_CHOICES = ("fp32", "fp16", "bf16", "amp")

_ALIASES = {
    "32": "fp32",
    "float32": "fp32",
    "fp32": "fp32",
    "16": "fp16",
    "float16": "fp16",
    "fp16": "fp16",
    "half": "fp16",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "amp": "amp",
}


@dataclass(frozen=True)
class PrecisionSpec:
    """Resolved autocast / GradScaler settings for one training run."""

    name: str
    resolved: str
    enabled: bool
    dtype: Optional[torch.dtype]
    use_scaler: bool
    device_type: str

    def autocast(self):
        return autocast(
            device_type=self.device_type,
            dtype=self.dtype if self.enabled else torch.float32,
            enabled=self.enabled,
        )

    def build_scaler(self) -> GradScaler:
        return GradScaler(self.device_type, enabled=self.use_scaler)

    def describe(self) -> str:
        if not self.enabled:
            return f"{self.name} (fp32, no autocast)"
        dtype = "bf16" if self.dtype == torch.bfloat16 else "fp16"
        scaler = " + GradScaler" if self.use_scaler else ""
        if self.name != self.resolved:
            return f"{self.name} -> {dtype} autocast{scaler}"
        return f"{self.resolved} autocast{scaler}"


def _normalize(name: str) -> str:
    key = str(name).strip().lower()
    if key not in _ALIASES:
        raise ValueError(
            f"Unknown precision {name!r}. Choose one of: {', '.join(PRECISION_CHOICES)}"
        )
    return _ALIASES[key]


def _bf16_supported(device: torch.device) -> bool:
    if device.type == "cuda":
        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    if device.type == "cpu":
        return True
    return False


def resolve_precision(name: str, device: torch.device) -> PrecisionSpec:
    """Map a user precision name to autocast dtype and whether fp16 scaling is needed.

    ``amp`` picks bf16 when the device supports it, otherwise fp16 + GradScaler.
    ``fp16`` always uses GradScaler. ``bf16`` does not.
    """
    requested = _normalize(name)
    device_type = device.type if device.type in ("cuda", "cpu", "mps") else "cpu"

    if requested == "fp32":
        return PrecisionSpec(requested, "fp32", False, torch.float32, False, device_type)

    resolved = requested
    if requested == "amp":
        resolved = "bf16" if _bf16_supported(device) else "fp16"

    if resolved == "bf16":
        if device.type == "cuda" and not _bf16_supported(device):
            raise RuntimeError(
                "precision=bf16 requires CUDA bf16 support; use fp16 or amp"
            )
        return PrecisionSpec(requested, "bf16", True, torch.bfloat16, False, device_type)

    if resolved == "fp16":
        if device_type == "cpu":
            return PrecisionSpec(requested, "bf16", True, torch.bfloat16, False, "cpu")
        return PrecisionSpec(requested, "fp16", True, torch.float16, True, device_type)

    raise ValueError(f"Unhandled precision {name!r}")
