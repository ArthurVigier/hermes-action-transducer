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


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert a HF/LeRobot robotics dataset to supervised ActionIR JSONL")
    ap.add_argument("--dataset-id", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--robot-profile", choices=["arm", "go2", "g1"], default="arm")
    ap.add_argument("--source-format", choices=["auto", "generic", "droid", "bridge"], default="auto")
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

    dataset = load_dataset(args.dataset_id, split=args.split, streaming=args.streaming)
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
    )
    save_jsonl_dataset(args.out, examples)
    print(
        {
            "dataset_id": args.dataset_id,
            "rows_read": len(rows),
            "episodes_written": len(examples),
            "out": args.out,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
