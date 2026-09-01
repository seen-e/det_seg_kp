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

from modules.config import config_from_args, parse_train_args
from datasets import build_dataloader
from modules.loss import build_criterion
from modules import DetSegKPModel
from modules.utils import PrecisionSpec, resolve_precision, set_seed
from modules.utils.checkpoint import save_checkpoint
from modules.utils.visualize import log_train_visualization, log_val_visualization
from modules.utils.distributed import (
    cleanup_distributed,
    init_wandb,
    is_main_process,
    log_info,
    reduce_dict,
    setup_distributed,
    unwrap_model,
)


class Trainer:
    """Owns training state so epoch / global_step / rank need not be threaded everywhere."""

    def __init__(self, args) -> None:
        self.rank, self.world_size, self.local_rank, self.distributed = setup_distributed(
            args.dist_backend
        )
        self.cfg = config_from_args(args)
        self.wandb_tags = args.wandb_tags

        self.epoch = 0
        self.global_step = 0

        self.device: torch.device
        self.precision: PrecisionSpec
        self.scaler: GradScaler
        self.model: torch.nn.Module
        self.criterion: torch.nn.Module
        self.optimizer: AdamW
        self.scheduler: SequentialLR
        self.ema: Optional[ModelEmaV2] = None
        self.train_loader = None
        self.val_loader = None
        self.wandb_run: Optional[Any] = None

        self.total_batch_size = 0
        self.lr_scale = 1.0
        self.base_lr = 0.0
        self.base_lr_backbone = 0.0
        self.warmup_epochs = 0
        self.cosine_epochs = 0

    @property
    def is_main(self) -> bool:
        return is_main_process(self.rank)

    def log(self, msg: str) -> None:
        log_info(self.rank, msg)

    def setup(self) -> None:
        set_seed(self.cfg.train.seed + self.rank)

        self.total_batch_size = self.cfg.train.batch_size * self.world_size
        self.lr_scale = math.sqrt(self.total_batch_size / float(LR_REF_BATCH_SIZE))
        self.base_lr, self.base_lr_backbone = self.cfg.train.lr, self.cfg.train.lr_backbone
        self.cfg.train.lr = self.base_lr * self.lr_scale
        self.cfg.train.lr_backbone = self.base_lr_backbone * self.lr_scale

        self.device = (
            torch.device("cuda", self.local_rank) if torch.cuda.is_available() else torch.device("cpu")
        )
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.precision = resolve_precision(self.cfg.train.precision, self.device)
        self.scaler = self.precision.build_scaler()

        if self.distributed:
            payload = [self.cfg.train.output_dir]
            dist.broadcast_object_list(payload, src=0)
            self.cfg.train.output_dir = payload[0]
        if self.is_main:
            os.makedirs(self.cfg.train.output_dir, exist_ok=True)
        if self.distributed:
            dist.barrier()

        self.wandb_run = (
            init_wandb(self.cfg, self.world_size, wandb_tags=self.wandb_tags)
            if self.is_main
            else None
        )

        self.log(
            f"Distributed: {self.distributed} | world_size={self.world_size} "
            f"| rank={self.rank} | local_rank={self.local_rank}"
        )
        self.log("Building dataloaders...")
        self.train_loader = build_dataloader(
            self.cfg, "train", self.cfg.train.batch_size, self.distributed
        )
        self.val_loader = build_dataloader(
            self.cfg, "val", self.cfg.train.batch_size, self.distributed
        )
        self.log(
            f"data_root={self.cfg.data.data_root} "
            f"val_ratio={self.cfg.data.val_ratio} "
            f"input={self.cfg.data.img_width}x{self.cfg.data.img_height} "
            f"stride={self.cfg.data.stride} "
            f"target={self.cfg.data.img_width // self.cfg.data.stride}"
            f"x{self.cfg.data.img_height // self.cfg.data.stride} "
            f"train={len(self.train_loader.dataset)} val={len(self.val_loader.dataset)}"
        )

        self.log("Building model...")
        model = DetSegKPModel(self.cfg.model).to(self.device)
        if self.distributed:
            model = DDP(model, device_ids=[self.local_rank] if self.device.type == "cuda" else None)
        self.model = model

        self.criterion = build_criterion(self.cfg).to(self.device)
        raw_model = unwrap_model(self.model)
        backbone_params, other_params = [], []
        for name, param in raw_model.named_parameters():
            if not param.requires_grad:
                continue
            (backbone_params if name.startswith("vision_tower") else other_params).append(param)
        self.optimizer = AdamW(
            [
                {"params": other_params, "lr": self.cfg.train.lr},
                {"params": backbone_params, "lr": self.cfg.train.lr_backbone},
            ],
            weight_decay=self.cfg.train.weight_decay,
        )
        self.warmup_epochs = int(round(self.cfg.train.epochs * self.cfg.train.warmup_epochs_ratio))
        self.warmup_epochs = max(0, min(self.warmup_epochs, max(self.cfg.train.epochs - 1, 0)))
        self.cosine_epochs = max(1, self.cfg.train.epochs - self.warmup_epochs)
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[
                LinearLR(
                    self.optimizer,
                    start_factor=1e-2 if self.warmup_epochs > 0 else 1.0,
                    end_factor=1.0,
                    total_iters=self.warmup_epochs,
                ),
                CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.cosine_epochs,
                    eta_min=self.cfg.train.lr * self.cfg.train.lr_min_ratio,
                ),
            ],
            milestones=[self.warmup_epochs],
        )
        self.ema = (
            ModelEmaV2(raw_model, decay=self.cfg.train.ema_decay, device=self.device)
            if self.cfg.train.use_ema
            else None
        )

        num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.log(f"Model parameters: {num_params / 1e6:.2f}M")
        self.log(f"Global batch size: {self.total_batch_size}")
        self.log(
            f"LR scale=sqrt({self.total_batch_size}/{LR_REF_BATCH_SIZE})={self.lr_scale:g} "
            f"(base_lr={self.base_lr:g} -> {self.cfg.train.lr:g}, "
            f"base_backbone_lr={self.base_lr_backbone:g} -> {self.cfg.train.lr_backbone:g})"
        )
        self.log(
            f"LR={self.cfg.train.lr} backbone_lr={self.cfg.train.lr_backbone} "
            f"{'warmup=' + str(self.warmup_epochs) + ' epochs then cosine' if self.warmup_epochs else 'cosine only'} "
            f"(T_max={self.cosine_epochs}, eta_min={self.cfg.train.lr * self.cfg.train.lr_min_ratio:g})"
        )
        self.log(f"Precision: {self.precision.describe()}")
        self.log(
            f"Val every {self.cfg.train.val_interval} epochs"
            if self.cfg.train.val_interval > 0
            else "Val disabled"
        )
        if self.ema is not None:
            self.log(f"EMA enabled (decay={self.cfg.train.ema_decay})")

    def train_one_epoch(self) -> Dict[str, float]:
        """One training epoch with AMP, optional EMA, and periodic GT/Pred dumps."""
        self.model.train()
        loss_accum: Dict[str, float] = {}
        num_steps = 0

        for step, batch in enumerate(self.train_loader):
            images = batch["images"].to(self.device, non_blocking=True)
            targets = [
                {k: v.to(self.device, non_blocking=True) for k, v in t.items()}
                for t in batch["targets"]
            ]

            with self.precision.autocast():
                outputs = self.model(images)
                loss, loss_dict = self.criterion(outputs, targets)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.cfg.train.clip_max_norm
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.ema is not None:
                self.ema.update(unwrap_model(self.model))

            num_steps += 1
            self.global_step += 1
            for k, v in loss_dict.items():
                loss_accum[k] = loss_accum.get(k, 0.0) + v

            if (
                self.cfg.train.vis_interval > 0
                and self.global_step % self.cfg.train.vis_interval == 0
                and self.is_main
            ):
                vis_path = log_train_visualization(
                    images[0],
                    targets[0],
                    outputs,
                    stride=self.cfg.data.stride,
                    output_dir=self.cfg.train.output_dir,
                    epoch=self.epoch,
                    step=self.global_step,
                    wandb_run=self.wandb_run,
                )
                self.log(f"  vis saved: {vis_path}")

            if (step + 1) % self.cfg.train.log_interval == 0:
                self.log(
                    f"  Epoch [{self.epoch}] Step [{step + 1}/{len(self.train_loader)}] "
                    + " ".join(f"{k}={v:.4f}" for k, v in loss_dict.items())
                )
                if self.wandb_run is not None and self.is_main:
                    self.wandb_run.log(
                        {f"train/{k}": v for k, v in loss_dict.items()}
                        | {"train/lr": self.optimizer.param_groups[0]["lr"]},
                        step=self.global_step,
                    )

        if num_steps == 0:
            return {}
        metrics = {k: v / num_steps for k, v in loss_accum.items()}
        return reduce_dict(metrics, self.world_size, self.device)

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Validation-loop loss average (no grad); may use EMA weights.

        Dumps every sample in this rank's shard to
        ``{output_dir}/vis_val/epoch_{epoch:05d}/``.
        """
        eval_model = self.ema.module if self.ema is not None else self.model
        eval_model.eval()
        loss_accum: Dict[str, float] = {}
        num_steps = 0
        vis_count = 0
        vis_dir = os.path.join(
            self.cfg.train.output_dir, "vis_val", f"epoch_{self.epoch:05d}"
        )

        for batch in self.val_loader:
            images = batch["images"].to(self.device, non_blocking=True)
            targets = [
                {k: v.to(self.device, non_blocking=True) for k, v in t.items()}
                for t in batch["targets"]
            ]
            with self.precision.autocast():
                outputs = eval_model(images)
                _, loss_dict = self.criterion(outputs, targets)
            stems = batch.get("stems") or [""] * images.shape[0]
            for i in range(images.shape[0]):
                stem = stems[i] if i < len(stems) else ""
                sample_name = stem or f"{vis_count:05d}"
                do_wandb = self.wandb_run is not None and self.is_main and vis_count == 0
                log_val_visualization(
                    images[i],
                    targets[i],
                    outputs,
                    stride=self.cfg.data.stride,
                    output_dir=self.cfg.train.output_dir,
                    epoch=self.epoch,
                    sample_name=sample_name,
                    sample_index=i,
                    wandb_run=self.wandb_run if do_wandb else None,
                    global_step=self.global_step,
                )
                vis_count += 1
            num_steps += 1
            for k, v in loss_dict.items():
                loss_accum[k] = loss_accum.get(k, 0.0) + v

        if self.is_main:
            self.log(f"  val vis saved under: {vis_dir} ({vis_count} samples on rank0)")

        if num_steps == 0:
            return {}
        metrics = {k: v / num_steps for k, v in loss_accum.items()}
        return reduce_dict(metrics, self.world_size, self.device)

    def run(self) -> None:
        self.setup()
        for self.epoch in range(1, self.cfg.train.epochs + 1):
            if self.distributed and hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(self.epoch)

            t0 = time.time()
            self.log(f"\n=== Epoch {self.epoch}/{self.cfg.train.epochs} ===")

            train_metrics = self.train_one_epoch()
            do_val = self.epoch == self.cfg.train.epochs or (
                self.cfg.train.val_interval > 0 and self.epoch % self.cfg.train.val_interval == 0
            )
            val_metrics: Dict[str, float] = self.evaluate() if do_val else {}
            self.scheduler.step()

            elapsed = time.time() - t0
            val_msg = f"Val loss={val_metrics.get('loss_total', 0.0):.4f} | " if val_metrics else ""
            self.log(
                f"Train loss={train_metrics.get('loss_total', 0.0):.4f} | {val_msg}Time={elapsed:.1f}s"
            )

            if self.wandb_run is not None and self.is_main:
                payload = {
                    **{f"epoch/train_{k}": v for k, v in train_metrics.items()},
                    "epoch/lr": self.optimizer.param_groups[0]["lr"],
                    "epoch": self.epoch,
                    "epoch/time_sec": elapsed,
                }
                if val_metrics:
                    payload.update({f"epoch/val_{k}": v for k, v in val_metrics.items()})
                self.wandb_run.log(payload, step=self.global_step)

            if self.is_main and self.epoch % self.cfg.train.save_interval == 0:
                ckpt_path = os.path.join(
                    self.cfg.train.output_dir, f"checkpoint_epoch{self.epoch}.pth"
                )
                save_checkpoint(
                    ckpt_path, self.epoch, self.model, self.optimizer, self.cfg, ema=self.ema
                )
                self.log(f"Saved checkpoint: {ckpt_path}")
                if self.wandb_run is not None and self.cfg.train.wandb_save_checkpoint:
                    self.wandb_run.save(
                        ckpt_path, base_path=self.cfg.train.output_dir, policy="now"
                    )

        if self.is_main:
            final_path = os.path.join(self.cfg.train.output_dir, "checkpoint_final.pth")
            save_checkpoint(
                final_path, self.cfg.train.epochs, self.model, self.optimizer, self.cfg, ema=self.ema
            )
            self.log(f"\nTraining complete. Final checkpoint: {final_path}")
            if self.wandb_run is not None:
                if self.cfg.train.wandb_save_checkpoint:
                    self.wandb_run.save(
                        final_path, base_path=self.cfg.train.output_dir, policy="now"
                    )
                self.wandb_run.finish()


def main() -> None:
    args = parse_train_args()
    try:
        Trainer(args).run()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
