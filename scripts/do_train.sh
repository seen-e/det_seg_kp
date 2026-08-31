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
#   DATA_ROOT        dataset root (default: ./data)
#   OUTPUT_DIR       checkpoint directory (default: ./work_dirs)
#   WANDB=0          disable Weights & Biases (default: on)
#
# Less common settings use --opt (see modules.config.Config):
#   bash scripts/do_train.sh --opt data.img_width=1280 --opt train.val_interval=500

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EPOCHS="${EPOCHS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-4}"
WARMUP_RATIO="${WARMUP_RATIO:-0}"
PRECISION="${PRECISION:-bf16}"
DATA_ROOT="${DATA_ROOT:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./work_dirs}"
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
  --data-root "$DATA_ROOT"
  --output-dir "$OUTPUT_DIR"
  --seed "$SEED"
  --dist-backend "$DIST_BACKEND"
  --opt "data.img_width=${IMG_WIDTH:-960}"
  --opt "data.img_height=${IMG_HEIGHT:-768}"
  --opt "data.stride=${STRIDE:-4}"
  --opt "data.num_workers=${NUM_WORKERS:-8}"
  --opt "model.object_query_decoder.num_queries=${NUM_QUERIES:-100}"
  --opt "train.log_interval=${LOG_INTERVAL:-10}"
  --opt "train.vis_interval=${VIS_INTERVAL:-100}"
  --opt "train.val_interval=${VAL_INTERVAL:-1000}"
  --opt "train.save_interval=${SAVE_INTERVAL:-5000}"
)

if [[ "${WANDB:-1}" == "1" ]]; then
  TRAIN_ARGS+=(--wandb --wandb-project "$WANDB_PROJECT")
  [[ -n "$WANDB_RUN_NAME" ]] && TRAIN_ARGS+=(--wandb-run-name "$WANDB_RUN_NAME")
  [[ -n "$WANDB_ENTITY" ]] && TRAIN_ARGS+=(--wandb-entity "$WANDB_ENTITY")
fi

TRAIN_ARGS+=("$@")

echo "Project root: $ROOT"
echo "GPUs (NPROC_PER_NODE): $NPROC_PER_NODE"
echo "Data root: $DATA_ROOT"
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
