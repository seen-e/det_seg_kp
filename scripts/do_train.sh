#!/usr/bin/env bash
# Launch Det+Seg+KP training (single-GPU or torchrun multi-GPU).
#
# Usage:
#   bash scripts/do_train.sh
#   bash scripts/do_train.sh --epochs 100 --batch-size 8
#   WANDB=0 bash scripts/do_train.sh
#
# Environment overrides (optional):
#   NPROC_PER_NODE   GPUs per node (default: auto-detect)
#   WARMUP_RATIO     LR warmup fraction of epochs (default: 0 = off)
#   DATA_ROOT        dataset root (default: ./data)
#   OUTPUT_DIR       checkpoint directory (default: ./outputs)
#   WANDB=0          disable Weights & Biases (default: on)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Defaults (override via env or pass extra flags through to train.py)
EPOCHS="${EPOCHS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0}"
PRECISION="${PRECISION:-bf16}"
IMG_WIDTH="${IMG_WIDTH:-960}"
IMG_HEIGHT="${IMG_HEIGHT:-768}"
STRIDE="${STRIDE:-4}"
DATA_ROOT="${DATA_ROOT:-./data}"
NUM_QUERIES="${NUM_QUERIES:-100}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LOG_INTERVAL="${LOG_INTERVAL:-10}"
VIS_INTERVAL="${VIS_INTERVAL:-100}"
VAL_INTERVAL="${VAL_INTERVAL:-1000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
SEED="${SEED:-42}"
DIST_BACKEND="${DIST_BACKEND:-nccl}"

WANDB_PROJECT="${WANDB_PROJECT:-det_seg_kp}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

if [[ -z "${NPROC_PER_NODE:-}" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC_PER_NODE="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    [[ "$NPROC_PER_NODE" -gt 0 ]] || NPROC_PER_NODE=1
  else
    NPROC_PER_NODE=1
  fi
fi

TRAIN_ARGS=(
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --lr "$LR"
  --warmup-ratio "$WARMUP_RATIO"
  --precision "$PRECISION"
  --img-width "$IMG_WIDTH"
  --img-height "$IMG_HEIGHT"
  --stride "$STRIDE"
  --data-root "$DATA_ROOT"
  --num-queries "$NUM_QUERIES"
  --num-workers "$NUM_WORKERS"
  --log-interval "$LOG_INTERVAL"
  --vis-interval "$VIS_INTERVAL"
  --val-interval "$VAL_INTERVAL"
  --save-interval "$SAVE_INTERVAL"
  --output-dir "$OUTPUT_DIR"
  --seed "$SEED"
  --dist-backend "$DIST_BACKEND"
)

if [[ "${WANDB:-1}" == "1" ]]; then
  TRAIN_ARGS+=(--wandb --wandb-project "$WANDB_PROJECT")
  [[ -n "$WANDB_RUN_NAME" ]] && TRAIN_ARGS+=(--wandb-run-name "$WANDB_RUN_NAME")
  [[ -n "$WANDB_ENTITY" ]] && TRAIN_ARGS+=(--wandb-entity "$WANDB_ENTITY")
fi

# Extra CLI args forwarded to train.py (e.g. --epochs 100)
TRAIN_ARGS+=("$@")

echo "Project root: $ROOT"
echo "GPUs (NPROC_PER_NODE): $NPROC_PER_NODE"
echo "Data root: $DATA_ROOT"
echo "Input: ${IMG_WIDTH}x${IMG_HEIGHT}  stride=${STRIDE}"
echo "Output dir: $OUTPUT_DIR"
echo "wandb: ${WANDB:-1}"

if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
  exec torchrun \
    --standalone \
    --nproc_per_node="$NPROC_PER_NODE" \
    scripts/train.py \
    "${TRAIN_ARGS[@]}"
else
  exec python scripts/train.py "${TRAIN_ARGS[@]}"
fi
