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

from hermes_action_transducer.dataset import JSONLSupervisedDataset
from hermes_action_transducer.features import FeatureConfig
from hermes_action_transducer.training import TrainingConfig, train_supervised


def main() -> int:
    ap = argparse.ArgumentParser(description="Train a supervised Hermes action transducer")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=96)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--feature-mode", choices=["compact", "rich", "per_layer"], default="rich")
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
    dataset = JSONLSupervisedDataset(args.dataset, feature_config=feature_config)
    metrics = train_supervised(
        dataset,
        args.checkpoint,
        config=TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
            device=args.device,
            feature_config=feature_config,
        ),
    )
    print(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "feature_config": feature_config.to_dict(),
                "feature_dim": dataset.feature_dim,
                "metrics": metrics,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
