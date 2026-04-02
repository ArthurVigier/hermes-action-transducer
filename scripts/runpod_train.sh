#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_PATH="${DATASET_PATH:-$ROOT_DIR/data/bootstrap_train.jsonl}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$ROOT_DIR/checkpoints/action_ir.pt}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
HIDDEN_DIM="${HIDDEN_DIM:-96}"
HF_DATASET_ID="${HF_DATASET_ID:-}"
HF_SPLIT="${HF_SPLIT:-train}"
ROBOT_PROFILE="${ROBOT_PROFILE:-arm}"
MAX_ROWS="${MAX_ROWS:-50000}"
MAX_EPISODES="${MAX_EPISODES:-2000}"
HF_STREAMING="${HF_STREAMING:-0}"
TASKS_JSONL="${TASKS_JSONL:-}"

echo "[runpod] root: $ROOT_DIR"
echo "[runpod] python: $PYTHON_BIN"

if [[ ! -d ".venv" ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .

python - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("torch") is None:
    print("Torch is not installed in this environment.", file=sys.stderr)
    print("Use a RunPod PyTorch image or install a CUDA-enabled torch build first.", file=sys.stderr)
    raise SystemExit(1)
PY

DEVICE="$("$PYTHON_BIN" - <<'PY'
import torch
print("cuda" if torch.cuda.is_available() else "cpu")
PY
)"

echo "[runpod] device: $DEVICE"

mkdir -p "$(dirname "$DATASET_PATH")" "$(dirname "$CHECKPOINT_PATH")"

if [[ ! -f "$DATASET_PATH" ]]; then
  if [[ -n "$HF_DATASET_ID" ]]; then
    echo "[runpod] installing conversion deps"
    python -m pip install -e ".[convert]"
    echo "[runpod] converting $HF_DATASET_ID -> $DATASET_PATH"
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
    "$PYTHON_BIN" scripts/convert_hf_dataset.py "${CONVERT_ARGS[@]}"
  else
    echo "[runpod] bootstrapping dataset -> $DATASET_PATH"
    "$PYTHON_BIN" scripts/bootstrap_dataset.py
  fi
fi

echo "[runpod] training checkpoint -> $CHECKPOINT_PATH"
"$PYTHON_BIN" scripts/train_supervised.py \
  --dataset "$DATASET_PATH" \
  --checkpoint "$CHECKPOINT_PATH" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --learning-rate "$LEARNING_RATE" \
  --hidden-dim "$HIDDEN_DIM" \
  --device "$DEVICE"

echo "[runpod] evaluating checkpoint -> $CHECKPOINT_PATH"
"$PYTHON_BIN" scripts/eval_supervised.py \
  --dataset "$DATASET_PATH" \
  --checkpoint "$CHECKPOINT_PATH" \
  --device "$DEVICE"

echo "[runpod] done"
