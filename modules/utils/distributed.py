"""Distributed training and logging helpers."""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from modules.config import Config


def setup_distributed(backend: str = "nccl") -> tuple[int, int, int, bool]:
    """Initialize torch.distributed when launched via torchrun."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0, False

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank % max(torch.cuda.device_count(), 1)))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    return rank, world_size, local_rank, True


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def reduce_dict(values: Dict[str, float], world_size: int, device: torch.device) -> Dict[str, float]:
    """Average scalar metrics across ranks."""
    if world_size <= 1:
        return values
    reduced: Dict[str, float] = {}
    for key, val in values.items():
        t = torch.tensor(val, dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        reduced[key] = float(t.item() / world_size)
    return reduced


def log_info(rank: int, msg: str) -> None:
    if is_main_process(rank):
        print(msg, flush=True)


def init_wandb(args: argparse.Namespace, cfg: Config, world_size: int) -> Optional[Any]:
    if not args.wandb:
        return None
    import wandb

    run_name = args.wandb_run_name or None
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=run_name,
        dir=cfg.train.output_dir,
        tags=args.wandb_tags,
        config={
            "epochs": cfg.train.epochs,
            "batch_size_per_gpu": cfg.train.batch_size,
            "world_size": world_size,
            "global_batch_size": cfg.train.batch_size * world_size,
            "lr": cfg.train.lr,
            "precision": cfg.train.precision,
            "img_width": cfg.data.img_width,
            "img_height": cfg.data.img_height,
            "num_queries": cfg.model.num_queries,
            "vision_tower": (
                asdict(cfg.model.vision_tower)
                if is_dataclass(cfg.model.vision_tower)
                else cfg.model.vision_tower
            ),
            "use_ema": cfg.train.use_ema,
            "ema_decay": cfg.train.ema_decay,
        },
    )
    return wandb
