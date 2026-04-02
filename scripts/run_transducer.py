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

from hermes_action_transducer.models import RobotObservation
from hermes_action_transducer.pipeline import ActionPipeline


def _parse_proprio(raw: str) -> list[float]:
    if not raw.strip():
        return []
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Hermes Action Transducer")
    ap.add_argument("--task", required=True)
    ap.add_argument("--state-text", default="")
    ap.add_argument("--image-path", default=None)
    ap.add_argument("--proprio", default="")
    ap.add_argument("--robot-profile", choices=["arm", "go2", "g1"], default="arm")
    args = ap.parse_args()

    observation = RobotObservation(
        task=args.task,
        state_text=args.state_text,
        image_path=args.image_path,
        proprio=_parse_proprio(args.proprio),
    )
    pipeline = ActionPipeline()
    result = pipeline.run(observation, robot_profile=args.robot_profile)
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
