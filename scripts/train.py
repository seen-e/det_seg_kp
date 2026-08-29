#!/usr/bin/env python3
"""Training script for det + seg + keypoint model.

Single GPU:
  python scripts/train.py --batch-size 4

Multi-GPU / multi-node (torchrun):
  torchrun --standalone --nproc_per_node=8 scripts/train.py --batch-size 4

  # two nodes, 8 GPUs each (run on every node with the same MASTER_ADDR/PORT):
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$NODE_RANK \\
      --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \\
      scripts/train.py --batch-size 4 --wandb --wandb-project det_seg_kp
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

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

from modules.config import Config
from datasets import build_train_dataloader
from modules.loss import build_criterion
from modules import DetSegKPModel
from modules.utils import PRECISION_CHOICES, PrecisionSpec, resolve_precision, set_seed
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Det+Seg+KP model")
    parser.add_argument("--epochs", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=4, help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--lr-backbone",
        type=float,
        default=None,
        help="Vision tower LR (default: 0.1 * --lr)",
    )
    parser.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.1,
        help="Fraction of epochs for linear LR warmup (0 disables warmup)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=list(PRECISION_CHOICES),
        help="Train dtype: fp32, fp16 (AMP+GradScaler), bf16 (AMP), amp (bf16 if supported else fp16)",
    )
    parser.add_argument("--img-width", type=int, default=960)
    parser.add_argument("--img-height", type=int, default=768)
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Mask / kp heatmap downsample vs input image (default 4)",
    )
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers per GPU")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--vis-interval",
        type=int,
        default=50,
        help="Dump GT/Pred visualization every N iters (0 to disable)",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=50,
        help="Run validation every N epochs (0 to disable; always runs on the last epoch)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5000,
        help="Save checkpoint every N epochs",
    )
    parser.add_argument("--output-dir", type=str, default="./work_dirs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dist-backend", type=str, default="nccl")

    parser.add_argument("--no-ema", action="store_true", help="Disable EMA weight averaging")
    parser.add_argument("--ema-decay", type=float, default=0.9999, help="EMA decay factor")

    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", type=str, default="det_seg_kp")
    parser.add_argument("--wandb-run-name", type=str, default="")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-tags", type=str, nargs="*", default=None)
    return parser.parse_args()


def train_one_epoch(
    model: DetSegKPModel,
    criterion,
    dataloader,
    optimizer,
    device: torch.device,
    epoch: int,
    rank: int,
    world_size: int,
    clip_max_norm: float,
    log_interval: int = 10,
    wandb_run: Optional[Any] = None,
    global_step: int = 0,
    ema: Optional[ModelEmaV2] = None,
    vis_interval: int = 50,
    output_dir: str = "./outputs",
    stride: int = 4,
    precision: Optional[PrecisionSpec] = None,
    scaler: Optional[GradScaler] = None,
) -> tuple[Dict[str, float], int]:
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_max_norm)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(unwrap_model(model))

        num_steps += 1
        global_step += 1
        for k, v in loss_dict.items():
            loss_accum[k] = loss_accum.get(k, 0.0) + v

        if vis_interval > 0 and global_step % vis_interval == 0 and is_main_process(rank):
            vis_path = log_train_visualization(
                images[0],
                targets[0],
                outputs,
                stride=stride,
                output_dir=output_dir,
                epoch=epoch,
                step=global_step,
                wandb_run=wandb_run,
            )
            log_info(rank, f"  vis saved: {vis_path}")

        if (step + 1) % log_interval == 0:
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
    args = parse_args()
    rank, world_size, local_rank, distributed = setup_distributed(args.dist_backend)

    try:
        set_seed(args.seed + rank)

        cfg = Config()
        cfg.train.epochs = args.epochs
        cfg.train.batch_size = args.batch_size
        cfg.train.lr = args.lr
        cfg.train.lr_backbone = (
            args.lr_backbone if args.lr_backbone is not None else args.lr * 0.1
        )
        cfg.train.warmup_epochs_ratio = args.warmup_ratio
        cfg.train.precision = args.precision
        cfg.train.output_dir = os.path.join(args.output_dir, time.strftime("%Y-%m-%d_%H-%M-%S"))
        cfg.train.device = args.device
        cfg.train.seed = args.seed
        cfg.train.use_wandb = args.wandb
        cfg.train.wandb_project = args.wandb_project
        cfg.train.wandb_run_name = args.wandb_run_name
        cfg.train.wandb_entity = args.wandb_entity
        cfg.data.img_width = args.img_width
        cfg.data.img_height = args.img_height
        cfg.data.data_root = args.data_root
        cfg.data.stride = args.stride
        cfg.data.num_workers = args.num_workers
        cfg.model.num_queries = args.num_queries
        if cfg.data.img_width % cfg.data.stride != 0 or cfg.data.img_height % cfg.data.stride != 0:
            raise ValueError(
                f"img_width={cfg.data.img_width}, img_height={cfg.data.img_height} "
                f"must be divisible by stride={cfg.data.stride}"
            )
        cfg.train.log_interval = args.log_interval
        cfg.train.vis_interval = args.vis_interval
        cfg.train.val_interval = args.val_interval
        cfg.train.save_interval = args.save_interval
        cfg.train.use_ema = not args.no_ema
        cfg.train.ema_decay = args.ema_decay

        if torch.cuda.is_available():
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
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

        wandb_run = None
        if is_main_process(rank):
            wandb_run = init_wandb(args, cfg, world_size)

        log_info(rank, f"Distributed: {distributed} | world_size={world_size} | rank={rank} | local_rank={local_rank}")
        log_info(rank, "Building dataloaders...")
        train_loader = build_train_dataloader(cfg, "train", cfg.train.batch_size, distributed, shuffle=True)
        val_loader = build_train_dataloader(cfg, "val", cfg.train.batch_size, distributed, shuffle=False)
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
            if name.startswith("vision_tower"):
                backbone_params.append(param)
            else:
                other_params.append(param)
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
        if warmup_epochs > 0:
            scheduler = SequentialLR(
                optimizer,
                schedulers=[
                    LinearLR(
                        optimizer,
                        start_factor=1e-2,
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
        else:
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=cosine_epochs,
                eta_min=cfg.train.lr * cfg.train.lr_min_ratio,
            )
        ema = (
            ModelEmaV2(raw_model, decay=cfg.train.ema_decay, device=device)
            if cfg.train.use_ema
            else None
        )

        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log_info(rank, f"Model parameters: {num_params / 1e6:.2f}M")
        log_info(rank, f"Global batch size: {cfg.train.batch_size * world_size}")
        if warmup_epochs > 0:
            log_info(
                rank,
                f"LR={cfg.train.lr} backbone_lr={cfg.train.lr_backbone} "
                f"warmup={warmup_epochs} epochs then cosine "
                f"(T_max={cosine_epochs}, eta_min={cfg.train.lr * cfg.train.lr_min_ratio:g})",
            )
        else:
            log_info(
                rank,
                f"LR={cfg.train.lr} backbone_lr={cfg.train.lr_backbone} "
                f"cosine only (T_max={cosine_epochs}, "
                f"eta_min={cfg.train.lr * cfg.train.lr_min_ratio:g})",
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
                cfg.train.clip_max_norm,
                cfg.train.log_interval,
                wandb_run=wandb_run,
                global_step=global_step,
                ema=ema,
                vis_interval=cfg.train.vis_interval,
                output_dir=cfg.train.output_dir,
                stride=cfg.data.stride,
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
            val_msg = (
                f"Val loss={val_metrics.get('loss_total', 0.0):.4f} | "
                if val_metrics
                else ""
            )
            log_info(
                rank,
                f"Train loss={train_metrics.get('loss_total', 0.0):.4f} | "
                f"{val_msg}"
                f"Time={elapsed:.1f}s",
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
