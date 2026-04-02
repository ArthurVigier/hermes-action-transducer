#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hermes_action_transducer.benchmark import run_feature_benchmark


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark multiple Hermes feature modes on the same dataset")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--results-out", required=True)
    ap.add_argument("--benchmark-mode", choices=["complete", "pair", "custom"], default="complete")
    ap.add_argument("--modes", default="")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=96)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rich-projection-dim", type=int, default=32)
    ap.add_argument("--layer-summary-dim", type=int, default=16)
    ap.add_argument("--per-layer-projection-dim", type=int, default=64)
    ap.add_argument("--max-layer-projections", type=int, default=3)
    ap.add_argument("--latency-samples", type=int, default=64)
    ap.add_argument("--latency-warmup", type=int, default=5)
    args = ap.parse_args()

    result = run_feature_benchmark(
        dataset_path=args.dataset,
        checkpoint_dir=args.checkpoint_dir,
        benchmark_mode=args.benchmark_mode,
        modes_csv=args.modes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        device=args.device,
        rich_projection_dim=args.rich_projection_dim,
        layer_summary_dim=args.layer_summary_dim,
        per_layer_projection_dim=args.per_layer_projection_dim,
        max_layer_projections=args.max_layer_projections,
        latency_samples=args.latency_samples,
        latency_warmup=args.latency_warmup,
    )
    out = Path(args.results_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
