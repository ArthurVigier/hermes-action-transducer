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
from hermes_action_transducer.training import evaluate_supervised


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate a supervised Hermes action transducer")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    dataset = JSONLSupervisedDataset(args.dataset)
    metrics = evaluate_supervised(dataset, args.checkpoint, device=args.device)
    print(json.dumps({"checkpoint": args.checkpoint, "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
