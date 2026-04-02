#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_MODE="${RUN_MODE:-train}"
DATASET_PATH="${DATASET_PATH:-$ROOT_DIR/data/bootstrap_train.jsonl}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$ROOT_DIR/checkpoints/action_ir.pt}"
FORCE_REBUILD_DATASET="${FORCE_REBUILD_DATASET:-0}"
FORCE_RETRAIN="${FORCE_RETRAIN:-0}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
HIDDEN_DIM="${HIDDEN_DIM:-96}"
BENCHMARK_MODE="${BENCHMARK_MODE:-complete}"
BENCHMARK_MODES="${BENCHMARK_MODES:-}"
BENCHMARK_RESULTS_PATH="${BENCHMARK_RESULTS_PATH:-$ROOT_DIR/benchmarks/results.json}"
BENCHMARK_CHECKPOINT_DIR="${BENCHMARK_CHECKPOINT_DIR:-$ROOT_DIR/benchmarks/checkpoints}"
LAYER_SWEEP_LAYERS="${LAYER_SWEEP_LAYERS:--1,-2,-4,-8,-16}"
LAYER_SWEEP_RESULTS_PATH="${LAYER_SWEEP_RESULTS_PATH:-$ROOT_DIR/benchmarks/layer_sweep_results.json}"
LAYER_SWEEP_DATASET_DIR="${LAYER_SWEEP_DATASET_DIR:-$ROOT_DIR/data/layer_sweep}"
LAYER_SWEEP_CHECKPOINT_DIR="${LAYER_SWEEP_CHECKPOINT_DIR:-$ROOT_DIR/benchmarks/layer_sweep_checkpoints}"
HF_DATASET_ID="${HF_DATASET_ID:-}"
HF_SPLIT="${HF_SPLIT:-train}"
ROBOT_PROFILE="${ROBOT_PROFILE:-arm}"
MAX_ROWS="${MAX_ROWS:-50000}"
MAX_EPISODES="${MAX_EPISODES:-2000}"
HF_STREAMING="${HF_STREAMING:-0}"
TASKS_JSONL="${TASKS_JSONL:-}"
FEATURE_MODE="${FEATURE_MODE:-rich}"
FEATURE_RICH_PROJECTION_DIM="${FEATURE_RICH_PROJECTION_DIM:-32}"
FEATURE_LAYER_SUMMARY_DIM="${FEATURE_LAYER_SUMMARY_DIM:-16}"
FEATURE_PER_LAYER_PROJECTION_DIM="${FEATURE_PER_LAYER_PROJECTION_DIM:-64}"
FEATURE_MAX_LAYER_PROJECTIONS="${FEATURE_MAX_LAYER_PROJECTIONS:-3}"
LATENCY_SAMPLES="${LATENCY_SAMPLES:-64}"
LATENCY_WARMUP="${LATENCY_WARMUP:-5}"
ENCODER_BACKEND="${ENCODER_BACKEND:-simple}"
HERMES_MODEL_ID="${HERMES_MODEL_ID:-NousResearch/Hermes-4.3-36B}"
HERMES_DEVICE_MAP="${HERMES_DEVICE_MAP:-auto}"
HERMES_TORCH_DTYPE="${HERMES_TORCH_DTYPE:-auto}"
HERMES_MAX_LENGTH="${HERMES_MAX_LENGTH:-1024}"
HERMES_LAYER_INDEX="${HERMES_LAYER_INDEX:--1}"
HERMES_POOL_STRATEGY="${HERMES_POOL_STRATEGY:-mean}"
HERMES_RICH_PROJECTION_DIM="${HERMES_RICH_PROJECTION_DIM:-128}"
HERMES_LAYER_PROJECTION_DIM="${HERMES_LAYER_PROJECTION_DIM:-64}"
HERMES_ADDITIONAL_LAYER_INDICES="${HERMES_ADDITIONAL_LAYER_INDICES:--4,-8}"
HERMES_ATTN_IMPLEMENTATION="${HERMES_ATTN_IMPLEMENTATION:-}"
HF_CACHE_DIR="${HF_CACHE_DIR:-}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-0}"

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

mkdir -p "$(dirname "$DATASET_PATH")" "$(dirname "$CHECKPOINT_PATH")" "$(dirname "$BENCHMARK_RESULTS_PATH")" "$BENCHMARK_CHECKPOINT_DIR"
if [[ -n "$HF_CACHE_DIR" ]]; then
  mkdir -p "$HF_CACHE_DIR"
  export HF_HOME="${HF_HOME:-$HF_CACHE_DIR}"
  export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_CACHE_DIR/hub}"
  mkdir -p "$HUGGINGFACE_HUB_CACHE"
fi

if [[ "$RUN_MODE" != "layer_sweep" && ( "$FORCE_REBUILD_DATASET" == "1" || ! -f "$DATASET_PATH" ) ]]; then
  if [[ -n "$HF_DATASET_ID" ]]; then
    echo "[runpod] installing conversion deps"
    python -m pip install -e ".[convert]"
    if [[ "$ENCODER_BACKEND" == "hermes_hf" ]]; then
      echo "[runpod] installing hermes encoder deps"
      python -m pip install -e ".[hermes]"
    fi
    echo "[runpod] converting $HF_DATASET_ID -> $DATASET_PATH"
    CONVERT_ARGS=(
      --dataset-id "$HF_DATASET_ID"
      --split "$HF_SPLIT"
      --robot-profile "$ROBOT_PROFILE"
      --encoder-backend "$ENCODER_BACKEND"
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
    if [[ "$ENCODER_BACKEND" == "hermes_hf" ]]; then
      CONVERT_ARGS+=(
        --model-id "$HERMES_MODEL_ID"
        --device-map "$HERMES_DEVICE_MAP"
        --torch-dtype "$HERMES_TORCH_DTYPE"
        --hf-cache-dir "$HF_CACHE_DIR"
        --max-length "$HERMES_MAX_LENGTH"
        --layer-index "$HERMES_LAYER_INDEX"
        --pool-strategy "$HERMES_POOL_STRATEGY"
        --rich-projection-dim "$HERMES_RICH_PROJECTION_DIM"
        --layer-projection-dim "$HERMES_LAYER_PROJECTION_DIM"
        "--additional-layer-indices=$HERMES_ADDITIONAL_LAYER_INDICES"
      )
      if [[ -n "$HERMES_ATTN_IMPLEMENTATION" ]]; then
        CONVERT_ARGS+=(--attn-implementation "$HERMES_ATTN_IMPLEMENTATION")
      fi
    fi
    if [[ -n "$HF_CACHE_DIR" && "$ENCODER_BACKEND" != "hermes_hf" ]]; then
      CONVERT_ARGS+=(--hf-cache-dir "$HF_CACHE_DIR")
    fi
    if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
      CONVERT_ARGS+=(--local-files-only)
    fi
    "$PYTHON_BIN" scripts/convert_hf_dataset.py "${CONVERT_ARGS[@]}"
  else
    echo "[runpod] bootstrapping dataset -> $DATASET_PATH"
    "$PYTHON_BIN" scripts/bootstrap_dataset.py
  fi
fi

if [[ "$RUN_MODE" == "benchmark" ]]; then
  echo "[runpod] running benchmark mode=$BENCHMARK_MODE modes=$BENCHMARK_MODES"
  "$PYTHON_BIN" scripts/benchmark_feature_modes.py \
    --dataset "$DATASET_PATH" \
    --checkpoint-dir "$BENCHMARK_CHECKPOINT_DIR" \
    --results-out "$BENCHMARK_RESULTS_PATH" \
    --benchmark-mode "$BENCHMARK_MODE" \
    --modes "$BENCHMARK_MODES" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --hidden-dim "$HIDDEN_DIM" \
    --rich-projection-dim "$FEATURE_RICH_PROJECTION_DIM" \
    --layer-summary-dim "$FEATURE_LAYER_SUMMARY_DIM" \
    --per-layer-projection-dim "$FEATURE_PER_LAYER_PROJECTION_DIM" \
    --max-layer-projections "$FEATURE_MAX_LAYER_PROJECTIONS" \
    --latency-samples "$LATENCY_SAMPLES" \
    --latency-warmup "$LATENCY_WARMUP" \
    --device "$DEVICE"
elif [[ "$RUN_MODE" == "layer_sweep" ]]; then
  echo "[runpod] running layer sweep layers=$LAYER_SWEEP_LAYERS"
  python -m pip install -e ".[convert,hermes]"
  LAYER_SWEEP_ARGS=(
    --dataset-id "$HF_DATASET_ID"
    --split "$HF_SPLIT"
    --robot-profile "$ROBOT_PROFILE"
    --model-id "$HERMES_MODEL_ID"
    --device-map "$HERMES_DEVICE_MAP"
    --torch-dtype "$HERMES_TORCH_DTYPE"
    --hf-cache-dir "$HF_CACHE_DIR"
    --max-length "$HERMES_MAX_LENGTH"
    --pool-strategy "$HERMES_POOL_STRATEGY"
    --rich-projection-dim "$HERMES_RICH_PROJECTION_DIM"
    --layer-projection-dim "$HERMES_LAYER_PROJECTION_DIM"
    "--additional-layer-indices=$HERMES_ADDITIONAL_LAYER_INDICES"
    --max-rows "$MAX_ROWS"
    --max-episodes "$MAX_EPISODES"
    "--layers=$LAYER_SWEEP_LAYERS"
    --dataset-dir "$LAYER_SWEEP_DATASET_DIR"
    --checkpoint-dir "$LAYER_SWEEP_CHECKPOINT_DIR"
    --results-out "$LAYER_SWEEP_RESULTS_PATH"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --learning-rate "$LEARNING_RATE"
    --hidden-dim "$HIDDEN_DIM"
    --feature-rich-projection-dim "$FEATURE_RICH_PROJECTION_DIM"
    --feature-layer-summary-dim "$FEATURE_LAYER_SUMMARY_DIM"
    --feature-per-layer-projection-dim "$FEATURE_PER_LAYER_PROJECTION_DIM"
    --feature-max-layer-projections "$FEATURE_MAX_LAYER_PROJECTIONS"
    --latency-samples "$LATENCY_SAMPLES"
    --latency-warmup "$LATENCY_WARMUP"
    --pair-modes "vanilla,compact"
    --device "$DEVICE"
  )
  if [[ "$FORCE_REBUILD_DATASET" == "1" ]]; then
    LAYER_SWEEP_ARGS+=(--force-rebuild-dataset)
  fi
  if [[ "$LOCAL_FILES_ONLY" == "1" ]]; then
    LAYER_SWEEP_ARGS+=(--local-files-only)
  fi
  if [[ "$HF_STREAMING" == "1" ]]; then
    LAYER_SWEEP_ARGS+=(--streaming)
  fi
  if [[ -n "$TASKS_JSONL" ]]; then
    LAYER_SWEEP_ARGS+=(--tasks-jsonl "$TASKS_JSONL")
  fi
  if [[ -n "$HERMES_ATTN_IMPLEMENTATION" ]]; then
    LAYER_SWEEP_ARGS+=(--attn-implementation "$HERMES_ATTN_IMPLEMENTATION")
  fi
  "$PYTHON_BIN" scripts/benchmark_layer_sweep.py "${LAYER_SWEEP_ARGS[@]}"
else
  if [[ "$FORCE_RETRAIN" == "1" || ! -f "$CHECKPOINT_PATH" ]]; then
    echo "[runpod] training checkpoint -> $CHECKPOINT_PATH"
    "$PYTHON_BIN" scripts/train_supervised.py \
      --dataset "$DATASET_PATH" \
      --checkpoint "$CHECKPOINT_PATH" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --learning-rate "$LEARNING_RATE" \
      --hidden-dim "$HIDDEN_DIM" \
      --feature-mode "$FEATURE_MODE" \
      --rich-projection-dim "$FEATURE_RICH_PROJECTION_DIM" \
      --layer-summary-dim "$FEATURE_LAYER_SUMMARY_DIM" \
      --per-layer-projection-dim "$FEATURE_PER_LAYER_PROJECTION_DIM" \
      --max-layer-projections "$FEATURE_MAX_LAYER_PROJECTIONS" \
      --device "$DEVICE"
  else
    echo "[runpod] reusing existing checkpoint -> $CHECKPOINT_PATH"
  fi

  echo "[runpod] evaluating checkpoint -> $CHECKPOINT_PATH"
  "$PYTHON_BIN" scripts/eval_supervised.py \
    --dataset "$DATASET_PATH" \
    --checkpoint "$CHECKPOINT_PATH" \
    --device "$DEVICE"
fi

echo "[runpod] done"
