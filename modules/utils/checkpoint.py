"""Checkpoint save/load helpers."""
from __future__ import annotations

from typing import Optional

import torch
from torch.optim import Optimizer
from timm.utils import ModelEmaV2

from modules.config import Config
from modules.utils.distributed import unwrap_model


def save_checkpoint(
    path: str,
    epoch: int,
    model: torch.nn.Module,
    optimizer: Optimizer,
    cfg: Config,
    ema: Optional[ModelEmaV2] = None,
) -> None:
    payload = {
        "epoch": epoch,
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": cfg,
    }
    if ema is not None:
        payload["ema"] = ema.module.state_dict()
    torch.save(payload, path)
