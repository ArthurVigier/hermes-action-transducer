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

from hermes_action_transducer.layer_sweep import parse_layer_indices_csv, run_layer_sweep


def main() -> int:
    ap = argparse.ArgumentParser(description="Sweep Hermes layer indices for compact-vs-vanilla benchmarking")
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--robot-profile", choices=["arm", "go2", "g1"], default="arm")
    ap.add_argument("--source-format", choices=["auto", "generic", "droid", "bridge"], default="auto")
    ap.add_argument("--model-id", default="NousResearch/Hermes-4.3-36B")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--torch-dtype", default="auto")
    ap.add_argument("--hf-cache-dir", default="")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--pool-strategy", choices=["mean", "last_token"], default="mean")
    ap.add_argument("--rich-projection-dim", type=int, default=128)
    ap.add_argument("--layer-projection-dim", type=int, default=64)
    ap.add_argument("--additional-layer-indices", default="-4,-8")
    ap.add_argument("--attn-implementation", default="")
    ap.add_argument("--max-rows", type=int, default=5000)
    ap.add_argument("--max-episodes", type=int, default=200)
    ap.add_argument("--tasks-jsonl", default="")
    ap.add_argument("--streaming", action="store_true")
    ap.add_argument("--layers", default="-1,-2,-4,-8,-16")
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--results-out", required=True)
    ap.add_argument("--force-rebuild-dataset", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=96)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--feature-rich-projection-dim", type=int, default=32)
    ap.add_argument("--feature-layer-summary-dim", type=int, default=16)
    ap.add_argument("--feature-per-layer-projection-dim", type=int, default=64)
    ap.add_argument("--feature-max-layer-projections", type=int, default=3)
    ap.add_argument("--latency-samples", type=int, default=64)
    ap.add_argument("--latency-warmup", type=int, default=5)
    ap.add_argument("--pair-modes", default="vanilla,compact")
    args = ap.parse_args()

    result = run_layer_sweep(
        root_dir=ROOT,
        dataset_id=args.dataset_id,
        split=args.split,
        robot_profile=args.robot_profile,
        source_format=args.source_format,
        model_id=args.model_id,
        device_map=args.device_map,
        torch_dtype=args.torch_dtype,
        hf_cache_dir=args.hf_cache_dir,
        local_files_only=args.local_files_only,
        max_length=args.max_length,
        pool_strategy=args.pool_strategy,
        rich_projection_dim=args.rich_projection_dim,
        layer_projection_dim=args.layer_projection_dim,
        additional_layer_indices=args.additional_layer_indices,
        attn_implementation=args.attn_implementation,
        max_rows=args.max_rows,
        max_episodes=args.max_episodes,
        tasks_jsonl=args.tasks_jsonl,
        streaming=args.streaming,
        layer_indices=parse_layer_indices_csv(args.layers),
        dataset_dir=args.dataset_dir,
        checkpoint_dir=args.checkpoint_dir,
        results_out=args.results_out,
        force_rebuild_dataset=args.force_rebuild_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        device=args.device,
        feature_rich_projection_dim=args.feature_rich_projection_dim,
        feature_layer_summary_dim=args.feature_layer_summary_dim,
        feature_per_layer_projection_dim=args.feature_per_layer_projection_dim,
        feature_max_layer_projections=args.feature_max_layer_projections,
        latency_samples=args.latency_samples,
        latency_warmup=args.latency_warmup,
        pair_modes_csv=args.pair_modes,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
