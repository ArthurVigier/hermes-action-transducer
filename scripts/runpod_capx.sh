#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONUNBUFFERED=1

PYTHON_BIN="${PYTHON_BIN:-python}"
CAPX_ROOT="${CAPX_ROOT:?CAPX_ROOT is required and should point to a capgym/cap-x checkout}"
CAPX_RESULTS_PATH="${CAPX_RESULTS_PATH:-$ROOT_DIR/benchmarks/capx_results.json}"
CAPX_LOG_DIR="${CAPX_LOG_DIR:-$ROOT_DIR/benchmarks/capx_logs}"
CAPX_SUITES="${CAPX_SUITES:-cube_stack}"
CAPX_BENCHMARK_MODE="${CAPX_BENCHMARK_MODE:-complete}"
CAPX_TIERS="${CAPX_TIERS:-}"
CAPX_CONFIGS="${CAPX_CONFIGS:-}"
CAPX_MODEL="${CAPX_MODEL:-NousResearch/Hermes-4.3-36B}"
CAPX_SERVER_URL="${CAPX_SERVER_URL:-http://127.0.0.1:8110/chat/completions}"
CAPX_TEMPERATURE="${CAPX_TEMPERATURE:-}"
CAPX_TOTAL_TRIALS="${CAPX_TOTAL_TRIALS:-}"
CAPX_NUM_WORKERS="${CAPX_NUM_WORKERS:-}"
CAPX_RECORD_VIDEO="${CAPX_RECORD_VIDEO:-0}"
CAPX_USE_ORACLE_CODE="${CAPX_USE_ORACLE_CODE:-0}"
CAPX_DRY_RUN="${CAPX_DRY_RUN:-0}"

echo "[runpod_capx] root: $ROOT_DIR"
echo "[runpod_capx] python: $PYTHON_BIN"
echo "[runpod_capx] capx_root: $CAPX_ROOT"

if [[ ! -d ".venv" ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .

mkdir -p "$(dirname "$CAPX_RESULTS_PATH")" "$CAPX_LOG_DIR"

ARGS=(
  --capx-root "$CAPX_ROOT"
  --results-out "$CAPX_RESULTS_PATH"
  --log-dir "$CAPX_LOG_DIR"
  --suites "$CAPX_SUITES"
  --benchmark-mode "$CAPX_BENCHMARK_MODE"
  --tiers "$CAPX_TIERS"
  --configs "$CAPX_CONFIGS"
  --model "$CAPX_MODEL"
  --server-url "$CAPX_SERVER_URL"
)

if [[ -n "$CAPX_TEMPERATURE" ]]; then
  ARGS+=(--temperature "$CAPX_TEMPERATURE")
fi
if [[ -n "$CAPX_TOTAL_TRIALS" ]]; then
  ARGS+=(--total-trials "$CAPX_TOTAL_TRIALS")
fi
if [[ -n "$CAPX_NUM_WORKERS" ]]; then
  ARGS+=(--num-workers "$CAPX_NUM_WORKERS")
fi
if [[ "$CAPX_RECORD_VIDEO" == "1" ]]; then
  ARGS+=(--record-video)
fi
if [[ "$CAPX_USE_ORACLE_CODE" == "1" ]]; then
  ARGS+=(--use-oracle-code)
fi
if [[ "$CAPX_DRY_RUN" == "1" ]]; then
  ARGS+=(--dry-run)
fi

python scripts/benchmark_capx.py "${ARGS[@]}"

echo "[runpod_capx] done"
