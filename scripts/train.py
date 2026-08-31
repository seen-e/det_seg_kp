#!/usr/bin/env python3
"""Training script for det + seg + keypoint model.

Single GPU:
  python scripts/train.py --batch-size 4

Multi-GPU / multi-node (torchrun):
  torchrun --standalone --nproc_per_node=8 scripts/train.py --batch-size 4

LR and backbone LR are scaled by ``sqrt(total_batch_size / 4)`` where
``total_batch_size = batch_size * world_size`` (reference batch size is 4).

Other config fields use :class:`modules.config.Config` defaults, or override via ``--opt``:
  python scripts/train.py --opt train.val_interval=1000 --opt data.img_width=960
"""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# LR is scaled by sqrt(total_batch_size / LR_REF_BATCH_SIZE).
LR_REF_BATCH_SIZE = 4

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from timm.utils import ModelEmaV2

from modules.config import Config, config_from_args, parse_train_args
from datasets import build_dataloader
from modules.loss import build_criterion
from modules import DetSegKPModel
from modules.utils import PrecisionSpec, resolve_precision, set_seed
from modules.utils.checkpoint import save_checkpoint
from modules.utils.visualize import log_train_visualization
from modules.utils.distributed import (
    cleanup_distributed,
    init_wandb,
    is_main_process,
    log_info,
    reduce_dict,
    setup_distributed,
    unwrap_model,
)


def train_one_epoch(
    model: DetSegKPModel,
    criterion,
    dataloader,
    optimizer,
    device: torch.device,
    epoch: int,
    rank: int,
    world_size: int,
    cfg: Config,
    wandb_run: Optional[Any] = None,
    global_step: int = 0,
    ema: Optional[ModelEmaV2] = None,
    precision: Optional[PrecisionSpec] = None,
    scaler: Optional[GradScaler] = None,
) -> tuple[Dict[str, float], int]:
    """One training epoch with AMP, optional EMA, and periodic GT/Pred dumps."""
    if precision is None:
        precision = resolve_precision("fp32", device)
    if scaler is None:
        scaler = precision.build_scaler()
    model.train()
    loss_accum: Dict[str, float] = {}
    num_steps = 0

    for step, batch in enumerate(dataloader):
        images = batch["images"].to(device, non_blocking=True)
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in batch["targets"]]

        with precision.autocast():
            outputs = model(images)
            loss, loss_dict = criterion(outputs, targets)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.train.clip_max_norm)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(unwrap_model(model))

        num_steps += 1
        global_step += 1
        for k, v in loss_dict.items():
            loss_accum[k] = loss_accum.get(k, 0.0) + v

        if cfg.train.vis_interval > 0 and global_step % cfg.train.vis_interval == 0 and is_main_process(rank):
            vis_path = log_train_visualization(
                images[0],
                targets[0],
                outputs,
                stride=cfg.data.stride,
                output_dir=cfg.train.output_dir,
                epoch=epoch,
                step=global_step,
                wandb_run=wandb_run,
            )
            log_info(rank, f"  vis saved: {vis_path}")

        if (step + 1) % cfg.train.log_interval == 0:
            log_info(
                rank,
                f"  Epoch [{epoch}] Step [{step+1}/{len(dataloader)}] "
                + " ".join(f"{k}={v:.4f}" for k, v in loss_dict.items()),
            )
            if wandb_run is not None and is_main_process(rank):
                wandb_run.log(
                    {f"train/{k}": v for k, v in loss_dict.items()} | {"train/lr": optimizer.param_groups[0]["lr"]},
                    step=global_step,
                )

    if num_steps == 0:
        return {}, global_step
    metrics = {k: v / num_steps for k, v in loss_accum.items()}
    return reduce_dict(metrics, world_size, device), global_step


@torch.no_grad()
def evaluate(
    model: DetSegKPModel,
    criterion,
    dataloader,
    device: torch.device,
    world_size: int,
    precision: Optional[PrecisionSpec] = None,
) -> Dict[str, float]:
    """Validation-loop loss average (no grad); caller may swap in EMA weights."""
    if precision is None:
        precision = resolve_precision("fp32", device)
    model.eval()
    loss_accum: Dict[str, float] = {}
    num_steps = 0

    for batch in dataloader:
        images = batch["images"].to(device, non_blocking=True)
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in batch["targets"]]
        with precision.autocast():
            outputs = model(images)
            _, loss_dict = criterion(outputs, targets)
        num_steps += 1
        for k, v in loss_dict.items():
            loss_accum[k] = loss_accum.get(k, 0.0) + v

    if num_steps == 0:
        return {}
    metrics = {k: v / num_steps for k, v in loss_accum.items()}
    return reduce_dict(metrics, world_size, device)


def main() -> None:
    args = parse_train_args()
    rank, world_size, local_rank, distributed = setup_distributed(args.dist_backend)

    try:
        cfg = config_from_args(args)
        set_seed(cfg.train.seed + rank)

        total_batch_size = cfg.train.batch_size * world_size
        lr_scale = math.sqrt(total_batch_size / float(LR_REF_BATCH_SIZE))
        base_lr, base_lr_backbone = cfg.train.lr, cfg.train.lr_backbone
        cfg.train.lr = base_lr * lr_scale
        cfg.train.lr_backbone = base_lr_backbone * lr_scale

        device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
        precision = resolve_precision(cfg.train.precision, device)
        scaler = precision.build_scaler()

        if distributed:
            payload = [cfg.train.output_dir]
            dist.broadcast_object_list(payload, src=0)
            cfg.train.output_dir = payload[0]
        if is_main_process(rank):
            os.makedirs(cfg.train.output_dir, exist_ok=True)
        if distributed:
            dist.barrier()

        wandb_run = init_wandb(cfg, world_size, wandb_tags=args.wandb_tags) if is_main_process(rank) else None

        log_info(rank, f"Distributed: {distributed} | world_size={world_size} | rank={rank} | local_rank={local_rank}")
        log_info(rank, "Building dataloaders...")
        train_loader = build_dataloader(cfg, "train", cfg.train.batch_size, distributed)
        val_loader = build_dataloader(cfg, "val", cfg.train.batch_size, distributed)
        log_info(
            rank,
            f"data_root={cfg.data.data_root} "
            f"input={cfg.data.img_width}x{cfg.data.img_height} "
            f"stride={cfg.data.stride} "
            f"target={cfg.data.img_width // cfg.data.stride}x{cfg.data.img_height // cfg.data.stride} "
            f"train={len(train_loader.dataset)} val={len(val_loader.dataset)}",
        )

        log_info(rank, "Building model...")
        model = DetSegKPModel(cfg.model).to(device)
        if distributed:
            model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

        criterion = build_criterion(cfg).to(device)
        raw_model = unwrap_model(model)
        backbone_params, other_params = [], []
        for name, param in raw_model.named_parameters():
            if not param.requires_grad:
                continue
            (backbone_params if name.startswith("vision_tower") else other_params).append(param)
        optimizer = AdamW(
            [
                {"params": other_params, "lr": cfg.train.lr},
                {"params": backbone_params, "lr": cfg.train.lr_backbone},
            ],
            weight_decay=cfg.train.weight_decay,
        )
        warmup_epochs = int(round(cfg.train.epochs * cfg.train.warmup_epochs_ratio))
        warmup_epochs = max(0, min(warmup_epochs, max(cfg.train.epochs - 1, 0)))
        cosine_epochs = max(1, cfg.train.epochs - warmup_epochs)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(
                    optimizer,
                    start_factor=1e-2 if warmup_epochs > 0 else 1.0,
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                ),
                CosineAnnealingLR(
                    optimizer,
                    T_max=cosine_epochs,
                    eta_min=cfg.train.lr * cfg.train.lr_min_ratio,
                ),
            ],
            milestones=[warmup_epochs],
        )
        ema = (
            ModelEmaV2(raw_model, decay=cfg.train.ema_decay, device=device)
            if cfg.train.use_ema
            else None
        )

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log_info(rank, f"Model parameters: {num_params / 1e6:.2f}M")
        log_info(rank, f"Global batch size: {total_batch_size}")
        log_info(
            rank,
            f"LR scale=sqrt({total_batch_size}/{LR_REF_BATCH_SIZE})={lr_scale:g} "
            f"(base_lr={base_lr:g} -> {cfg.train.lr:g}, "
            f"base_backbone_lr={base_lr_backbone:g} -> {cfg.train.lr_backbone:g})",
        )
        log_info(
            rank,
            f"LR={cfg.train.lr} backbone_lr={cfg.train.lr_backbone} "
            f"{'warmup=' + str(warmup_epochs) + ' epochs then cosine' if warmup_epochs else 'cosine only'} "
            f"(T_max={cosine_epochs}, eta_min={cfg.train.lr * cfg.train.lr_min_ratio:g})",
        )
        log_info(rank, f"Precision: {precision.describe()}")
        log_info(rank, f"Val every {cfg.train.val_interval} epochs" if cfg.train.val_interval > 0 else "Val disabled")
        if ema is not None:
            log_info(rank, f"EMA enabled (decay={cfg.train.ema_decay})")

        global_step = 0
        for epoch in range(1, cfg.train.epochs + 1):
            if distributed and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            t0 = time.time()
            log_info(rank, f"\n=== Epoch {epoch}/{cfg.train.epochs} ===")

            train_metrics, global_step = train_one_epoch(
                model,
                criterion,
                train_loader,
                optimizer,
                device,
                epoch,
                rank,
                world_size,
                cfg,
                wandb_run=wandb_run,
                global_step=global_step,
                ema=ema,
                precision=precision,
                scaler=scaler,
            )
            do_val = (
                epoch == cfg.train.epochs
                or (cfg.train.val_interval > 0 and epoch % cfg.train.val_interval == 0)
            )
            val_metrics: Dict[str, float] = {}
            if do_val:
                eval_model = ema.module if ema is not None else model
                val_metrics = evaluate(
                    eval_model, criterion, val_loader, device, world_size, precision=precision
                )
            scheduler.step()

            elapsed = time.time() - t0
            val_msg = f"Val loss={val_metrics.get('loss_total', 0.0):.4f} | " if val_metrics else ""
            log_info(
                rank,
                f"Train loss={train_metrics.get('loss_total', 0.0):.4f} | {val_msg}Time={elapsed:.1f}s",
            )

            if wandb_run is not None and is_main_process(rank):
                payload = {
                    **{f"epoch/train_{k}": v for k, v in train_metrics.items()},
                    "epoch/lr": optimizer.param_groups[0]["lr"],
                    "epoch": epoch,
                    "epoch/time_sec": elapsed,
                }
                if val_metrics:
                    payload.update({f"epoch/val_{k}": v for k, v in val_metrics.items()})
                wandb_run.log(payload, step=global_step)

            if is_main_process(rank) and epoch % cfg.train.save_interval == 0:
                ckpt_path = os.path.join(cfg.train.output_dir, f"checkpoint_epoch{epoch}.pth")
                save_checkpoint(ckpt_path, epoch, model, optimizer, cfg, ema=ema)
                log_info(rank, f"Saved checkpoint: {ckpt_path}")
                if wandb_run is not None:
                    wandb_run.save(ckpt_path, base_path=cfg.train.output_dir, policy="now")

        if is_main_process(rank):
            final_path = os.path.join(cfg.train.output_dir, "checkpoint_final.pth")
            save_checkpoint(final_path, cfg.train.epochs, model, optimizer, cfg, ema=ema)
            log_info(rank, f"\nTraining complete. Final checkpoint: {final_path}")
            if wandb_run is not None:
                wandb_run.save(final_path, base_path=cfg.train.output_dir, policy="now")
                wandb_run.finish()

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
