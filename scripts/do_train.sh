#!/usr/bin/env bash
# Launch Det+Seg+KP training (single-GPU or torchrun multi-GPU).
#
# Usage:
#   bash scripts/do_train.sh
#   bash scripts/do_train.sh --epochs 100 --batch-size 8
#   WANDB=0 bash scripts/do_train.sh
#   VAL_RATIO=0.1 IMG_WIDTH=1280 bash scripts/do_train.sh
#
# Environment overrides (optional) — all defaults live in TRAIN_ARGS below:
#   NPROC_PER_NODE, EPOCHS, BATCH_SIZE, LR, WARMUP_RATIO, PRECISION,
#   DATA_ROOT, OUTPUT_DIR, SEED, DIST_BACKEND, EMA_DECAY, VAL_RATIO,
#   IMG_WIDTH, IMG_HEIGHT, STRIDE, KP_SIGMA, KP_THRESHOLD, NUM_WORKERS, PREFETCH_FACTOR,
#   NUM_QUERIES, LOG_INTERVAL, VIS_INTERVAL, VAL_INTERVAL, SAVE_INTERVAL,
#   WANDB=0, WANDB_PROJECT, WANDB_RUN_NAME, WANDB_ENTITY, WANDB_SAVE_CKPT=1
#
# Extra CLI flags pass through:
#   bash scripts/do_train.sh --opt train.clip_max_norm=0.05

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC_PER_NODE="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    [[ "$NPROC_PER_NODE" -gt 0 ]] || NPROC_PER_NODE=1
  else
    NPROC_PER_NODE=1
  fi
fi

TRAIN_ARGS=(
  --epochs "${EPOCHS:-300}"
  --batch-size "${BATCH_SIZE:-16}"
  --lr "${LR:-1e-4}"
  --warmup-ratio "${WARMUP_RATIO:-0.0}"
  --precision "${PRECISION:-bf16}"
  --data-root "${DATA_ROOT:-/mnt/data/Det_Seg_KP}"
  --output-dir "${OUTPUT_DIR:-./work_dirs}"
  --seed "${SEED:-42}"
  --dist-backend "${DIST_BACKEND:-nccl}"
  --ema-decay "${EMA_DECAY:-0.999}"
  --opt "data.img_width=${IMG_WIDTH:-960}"
  --opt "data.img_height=${IMG_HEIGHT:-768}"
  --opt "data.stride=${STRIDE:-4}"
  --opt "data.kp_sigma=${KP_SIGMA:-1.2}"
  --opt "data.kp_threshold=${KP_THRESHOLD:-0.01}"
  --opt "data.val_ratio=${VAL_RATIO:-0.01}"
  --opt "data.num_workers=${NUM_WORKERS:-8}"
  --opt "data.prefetch_factor=${PREFETCH_FACTOR:-2}"
  --opt "model.object_query_decoder.num_queries=${NUM_QUERIES:-100}"
  --opt "train.log_interval=${LOG_INTERVAL:-10}"
  --opt "train.vis_interval=${VIS_INTERVAL:-100}"
  --opt "train.val_interval=${VAL_INTERVAL:-10}"
  --opt "train.save_interval=${SAVE_INTERVAL:-50}"
)

if [[ "${WANDB:-1}" == "1" ]]; then
  TRAIN_ARGS+=(--wandb --wandb-project "${WANDB_PROJECT:-det_seg_kp}")
  [[ -n "${WANDB_RUN_NAME:-}" ]] && TRAIN_ARGS+=(--wandb-run-name "$WANDB_RUN_NAME")
  [[ -n "${WANDB_ENTITY:-}" ]] && TRAIN_ARGS+=(--wandb-entity "$WANDB_ENTITY")
  if [[ "${WANDB_SAVE_CKPT:-0}" != "1" ]]; then
    TRAIN_ARGS+=(--no-wandb-save-checkpoint)
  fi
fi

TRAIN_ARGS+=("$@")

echo "Project root: $ROOT"
echo "GPUs (NPROC_PER_NODE): $NPROC_PER_NODE"
echo "Data root: ${DATA_ROOT:-/mnt/data/Det_Seg_KP}"
echo "Output dir: ${OUTPUT_DIR:-./work_dirs}"
echo "EMA decay: ${EMA_DECAY:-0.999}"
echo "Val ratio: ${VAL_RATIO:-0.01}"
echo "KP sigma: ${KP_SIGMA:-1.2}  threshold: ${KP_THRESHOLD:-0.01}"
echo "wandb: ${WANDB:-1}"
echo "wandb save ckpt: ${WANDB_SAVE_CKPT:-0}"

if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
  exec torchrun \
    --standalone \
    --nproc_per_node="$NPROC_PER_NODE" \
    scripts/train.py \
    "${TRAIN_ARGS[@]}"
else
  exec python scripts/train.py "${TRAIN_ARGS[@]}"
fi
