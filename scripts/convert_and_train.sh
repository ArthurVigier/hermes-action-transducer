#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
HF_DATASET_ID="${HF_DATASET_ID:?HF_DATASET_ID is required}"
HF_SPLIT="${HF_SPLIT:-train}"
ROBOT_PROFILE="${ROBOT_PROFILE:-arm}"
DATASET_PATH="${DATASET_PATH:-$ROOT_DIR/data/${ROBOT_PROFILE}_supervised.jsonl}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$ROOT_DIR/checkpoints/action_ir.pt}"
MAX_ROWS="${MAX_ROWS:-50000}"
MAX_EPISODES="${MAX_EPISODES:-2000}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
HIDDEN_DIM="${HIDDEN_DIM:-96}"
DEVICE="${DEVICE:-cpu}"
HF_STREAMING="${HF_STREAMING:-0}"
TASKS_JSONL="${TASKS_JSONL:-}"

mkdir -p "$(dirname "$DATASET_PATH")" "$(dirname "$CHECKPOINT_PATH")"

CONVERT_ARGS=(
  --dataset-id "$HF_DATASET_ID"
  --split "$HF_SPLIT"
  --robot-profile "$ROBOT_PROFILE"
  --out "$DATASET_PATH"
  --max-rows "$MAX_ROWS"
  --max-episodes "$MAX_EPISODES"
)

if [[ "$HF_STREAMING" == "1" ]]; then
  CONVERT_ARGS+=(--streaming)
fi

if [[ -n "$TASKS_JSONL" ]]; then
  CONVERT_ARGS+=(--tasks-jsonl "$TASKS_JSONL")
fi

echo "[convert_and_train] converting $HF_DATASET_ID -> $DATASET_PATH"
"$PYTHON_BIN" scripts/convert_hf_dataset.py "${CONVERT_ARGS[@]}"

echo "[convert_and_train] training -> $CHECKPOINT_PATH"
"$PYTHON_BIN" scripts/train_supervised.py \
  --dataset "$DATASET_PATH" \
  --checkpoint "$CHECKPOINT_PATH" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --hidden-dim "$HIDDEN_DIM" \
  --device "$DEVICE"

echo "[convert_and_train] evaluating -> $CHECKPOINT_PATH"
"$PYTHON_BIN" scripts/eval_supervised.py \
  --dataset "$DATASET_PATH" \
  --checkpoint "$CHECKPOINT_PATH" \
  --device "$DEVICE"
