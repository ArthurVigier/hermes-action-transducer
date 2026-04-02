#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hermes_action_transducer.conversion import (
    convert_episode_rows_to_supervised_rows,
    load_task_lookup_from_jsonl,
)
from hermes_action_transducer.data_io import save_jsonl_dataset


def _parse_layer_indices(raw: str) -> tuple[int, ...]:
    if not raw.strip():
        return ()
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())
from hermes_action_transducer.encoder import HermesHFConfig, HermesHFEncoder


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert a HF/LeRobot robotics dataset to supervised ActionIR JSONL")
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--robot-profile", choices=["arm", "go2", "g1"], default="arm")
    ap.add_argument("--source-format", choices=["auto", "generic", "droid", "bridge"], default="auto")
    ap.add_argument("--encoder-backend", choices=["simple", "hermes_hf"], default="simple")
    ap.add_argument("--model-id", default="NousResearch/Hermes-4.3-36B")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--torch-dtype", default="auto")
    ap.add_argument("--hf-cache-dir", default=None)
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--layer-index", type=int, default=-1)
    ap.add_argument("--pool-strategy", choices=["mean", "last_token"], default="mean")
    ap.add_argument("--rich-projection-dim", type=int, default=128)
    ap.add_argument("--layer-projection-dim", type=int, default=64)
    ap.add_argument("--additional-layer-indices", default="-4,-8")
    ap.add_argument("--attn-implementation", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-rows", type=int, default=50000)
    ap.add_argument("--max-episodes", type=int, default=2000)
    ap.add_argument("--streaming", action="store_true")
    ap.add_argument("--tasks-jsonl", default=None)
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install optional deps first: pip install -e '.[convert]'") from exc

    task_lookup = {}
    if args.tasks_jsonl:
        task_lookup = load_task_lookup_from_jsonl(args.tasks_jsonl)

    encoder = None
    if args.encoder_backend == "hermes_hf":
        encoder = HermesHFEncoder(
            HermesHFConfig(
                model_id=args.model_id,
                device_map=args.device_map,
                torch_dtype=args.torch_dtype,
                cache_dir=args.hf_cache_dir,
                local_files_only=args.local_files_only,
                max_length=args.max_length,
                layer_index=args.layer_index,
                pool_strategy=args.pool_strategy,
                rich_projection_dim=args.rich_projection_dim,
                layer_projection_dim=args.layer_projection_dim,
                additional_layer_indices=_parse_layer_indices(args.additional_layer_indices),
                attn_implementation=args.attn_implementation,
            )
        )

    load_dataset_kwargs = {
        "split": args.split,
        "streaming": args.streaming,
    }
    if args.hf_cache_dir:
        load_dataset_kwargs["cache_dir"] = args.hf_cache_dir
    if args.local_files_only:
        load_dataset_kwargs["download_mode"] = "reuse_dataset_if_exists"
    dataset = load_dataset(args.dataset_id, **load_dataset_kwargs)
    rows = []
    for idx, row in enumerate(dataset):
        payload = dict(row)
        payload["_dataset_id"] = args.dataset_id
        rows.append(payload)
        if idx + 1 >= args.max_rows:
            break

    examples = convert_episode_rows_to_supervised_rows(
        rows,
        robot_profile=args.robot_profile,
        task_lookup=task_lookup or None,
        max_episodes=args.max_episodes,
        source_format=args.source_format,
        encoder=encoder,
    )
    save_jsonl_dataset(args.out, examples)
    print(
        {
            "dataset_id": args.dataset_id,
            "rows_read": len(rows),
            "episodes_written": len(examples),
            "out": args.out,
            "encoder_backend": args.encoder_backend,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
