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

from hermes_action_transducer.capx import run_capx_benchmark


def main() -> int:
    ap = argparse.ArgumentParser(description="Run or dry-run a CapX benchmark from this repo")
    ap.add_argument("--capx-root", required=True, help="Path to the official capgym/cap-x checkout")
    ap.add_argument("--results-out", required=True)
    ap.add_argument("--log-dir", default="")
    ap.add_argument("--suites", default="cube_stack")
    ap.add_argument("--benchmark-mode", choices=["complete", "pair", "custom"], default="complete")
    ap.add_argument("--tiers", default="")
    ap.add_argument("--configs", default="")
    ap.add_argument("--model", default="NousResearch/Hermes-4.3-36B")
    ap.add_argument("--server-url", default="http://127.0.0.1:8110/chat/completions")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--total-trials", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--record-video", action="store_true")
    ap.add_argument("--use-oracle-code", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    result = run_capx_benchmark(
        capx_root=args.capx_root,
        suites_csv=args.suites,
        benchmark_mode=args.benchmark_mode,
        tiers_csv=args.tiers,
        model=args.model,
        server_url=args.server_url,
        results_out=args.results_out,
        log_dir=args.log_dir or None,
        total_trials=args.total_trials,
        num_workers=args.num_workers,
        temperature=args.temperature,
        record_video=args.record_video,
        use_oracle_code=args.use_oracle_code,
        dry_run=args.dry_run,
        configs_csv=args.configs,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
