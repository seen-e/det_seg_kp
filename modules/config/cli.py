"""CLI parsing and applying overrides onto :class:`Config`."""
from __future__ import annotations

import argparse
import os
import time
from typing import Any, Iterable, Sequence, Type

from modules.utils import PRECISION_CHOICES

from .config import Config


def _set_nested(obj: Any, path: str, value: Any) -> None:
    parts = path.split(".")
    for key in parts[:-1]:
        obj = getattr(obj, key)
    setattr(obj, parts[-1], value)


def _parse_opt_value(raw: str) -> Any:
    lowered = raw.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered == "none":
        return None
    try:
        if "." in raw or "e" in lowered:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw


def apply_opts(cfg: Config, opts: Iterable[str]) -> None:
    """Apply ``section.field=value`` overrides (repeatable ``--opt``)."""
    for item in opts:
        if "=" not in item:
            raise ValueError(f"--opt expects key=value, got {item!r}")
        key, _, raw = item.partition("=")
        key = key.strip()
        if not key:
            raise ValueError(f"--opt expects key=value, got {item!r}")
        _set_nested(cfg, key, _parse_opt_value(raw.strip()))


def _add_opt(
    parser: argparse.ArgumentParser,
    *flags: str,
    dest: str,
    type_: Type,
    cfg: Config,
    path: str,
    help_: str = "",
) -> None:
    default_val = _get_nested(cfg, path.split("."))
    hint = help_ or f"Config `{path}` (default: {default_val})"
    parser.add_argument(*flags, dest=dest, type=type_, default=argparse.SUPPRESS, help=hint)


def _get_nested(obj: Any, parts: Sequence[str]) -> Any:
    for key in parts:
        obj = getattr(obj, key)
    return obj


def build_train_parser(cfg: Config | None = None) -> argparse.ArgumentParser:
    """Build argparse. Unspecified flags keep :class:`Config` defaults."""
    cfg = cfg or Config()
    p = argparse.ArgumentParser(description="Train Det+Seg+KP model")

    _add_opt(p, "--epochs", dest="epochs", type_=int, cfg=cfg, path="train.epochs")
    _add_opt(p, "--batch-size", dest="batch_size", type_=int, cfg=cfg, path="train.batch_size")
    _add_opt(p, "--lr", dest="lr", type_=float, cfg=cfg, path="train.lr")
    _add_opt(
        p,
        "--lr-backbone",
        dest="lr_backbone",
        type_=float,
        cfg=cfg,
        path="train.lr_backbone",
        help_="Vision tower LR (default: 0.1 * lr when only --lr is set)",
    )
    _add_opt(
        p,
        "--warmup-ratio",
        dest="warmup_ratio",
        type_=float,
        cfg=cfg,
        path="train.warmup_epochs_ratio",
    )
    _add_opt(p, "--precision", dest="precision", type_=str, cfg=cfg, path="train.precision",
             help_=f"One of {', '.join(PRECISION_CHOICES)}")
    _add_opt(p, "--output-dir", dest="output_dir", type_=str, cfg=cfg, path="train.output_dir")
    _add_opt(p, "--seed", dest="seed", type_=int, cfg=cfg, path="train.seed")

    _add_opt(p, "--data-root", dest="data_root", type_=str, cfg=cfg, path="data.data_root")

    p.add_argument(
        "--opt",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override any config field, e.g. train.val_interval=1000 data.stride=4",
    )
    p.add_argument(
        "--dist-backend",
        default="nccl",
        help="torch.distributed backend (default: nccl)",
    )

    p.add_argument("--no-ema", action="store_true", help="Disable EMA")
    _add_opt(p, "--ema-decay", dest="ema_decay", type_=float, cfg=cfg, path="train.ema_decay")

    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases")
    _add_opt(p, "--wandb-project", dest="wandb_project", type_=str, cfg=cfg, path="train.wandb_project")
    _add_opt(p, "--wandb-run-name", dest="wandb_run_name", type_=str, cfg=cfg, path="train.wandb_run_name")
    _add_opt(p, "--wandb-entity", dest="wandb_entity", type_=str, cfg=cfg, path="train.wandb_entity")
    p.add_argument(
        "--no-wandb-save-checkpoint",
        action="store_true",
        help="Do not upload checkpoints to wandb (metrics/images still logged)",
    )
    p.add_argument("--wandb-tags", nargs="*", default=None, help="Optional wandb tags")

    return p


def config_from_args(args: argparse.Namespace, base: Config | None = None) -> Config:
    """Merge explicit CLI flags and ``--opt`` overrides into a :class:`Config`."""
    cfg = base or Config()
    explicit = {
        k: v
        for k, v in vars(args).items()
        if k not in ("opt", "dist_backend", "wandb_tags", "no_wandb_save_checkpoint")
    }

    _FIELD_MAP = {
        "epochs": "train.epochs",
        "batch_size": "train.batch_size",
        "lr": "train.lr",
        "lr_backbone": "train.lr_backbone",
        "warmup_ratio": "train.warmup_epochs_ratio",
        "precision": "train.precision",
        "output_dir": "train.output_dir",
        "seed": "train.seed",
        "data_root": "data.data_root",
        "ema_decay": "train.ema_decay",
        "wandb_project": "train.wandb_project",
        "wandb_run_name": "train.wandb_run_name",
        "wandb_entity": "train.wandb_entity",
    }

    for dest, path in _FIELD_MAP.items():
        if dest in explicit:
            _set_nested(cfg, path, explicit[dest])

    if "lr" in explicit and "lr_backbone" not in explicit:
        cfg.train.lr_backbone = cfg.train.lr * 0.1

    if getattr(args, "no_ema", False):
        cfg.train.use_ema = False
    if getattr(args, "wandb", False):
        cfg.train.use_wandb = True
    if getattr(args, "no_wandb_save_checkpoint", False):
        cfg.train.wandb_save_checkpoint = False

    apply_opts(cfg, args.opt)

    if cfg.train.precision not in PRECISION_CHOICES:
        raise ValueError(f"precision must be one of {PRECISION_CHOICES}, got {cfg.train.precision!r}")

    stamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    cfg.train.output_dir = os.path.join(cfg.train.output_dir, stamp)

    if cfg.data.img_width % cfg.data.stride != 0 or cfg.data.img_height % cfg.data.stride != 0:
        raise ValueError(
            f"img_width={cfg.data.img_width}, img_height={cfg.data.img_height} "
            f"must be divisible by stride={cfg.data.stride}"
        )
    return cfg


def parse_train_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_train_parser().parse_args(argv)
