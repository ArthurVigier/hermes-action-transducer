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

from hermes_action_transducer.features import FeatureConfig
from hermes_action_transducer.plan_stack import PlanStackTrainingConfig, train_plan_stack


def main() -> int:
    ap = argparse.ArgumentParser(description="Train a hierarchical Hermes plan stack")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--results-out", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=96)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-plan-codes", type=int, default=8)
    ap.add_argument("--kmeans-iterations", type=int, default=12)
    ap.add_argument("--feature-mode", choices=["vanilla", "compact", "rich", "per_layer", "full"], default="compact")
    ap.add_argument("--rich-projection-dim", type=int, default=32)
    ap.add_argument("--layer-summary-dim", type=int, default=16)
    ap.add_argument("--per-layer-projection-dim", type=int, default=64)
    ap.add_argument("--max-layer-projections", type=int, default=3)
    args = ap.parse_args()

    feature_config = FeatureConfig(
        mode=args.feature_mode,
        rich_projection_dim=args.rich_projection_dim,
        layer_summary_dim=args.layer_summary_dim,
        per_layer_projection_dim=args.per_layer_projection_dim,
        max_layer_projections=args.max_layer_projections,
    )
    results = train_plan_stack(
        args.dataset,
        args.checkpoint_dir,
        config=PlanStackTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
            device=args.device,
            feature_config=feature_config,
            max_plan_codes=args.max_plan_codes,
            kmeans_iterations=args.kmeans_iterations,
        ),
    )
    out = Path(args.results_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
