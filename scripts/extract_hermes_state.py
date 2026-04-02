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

from hermes_action_transducer.encoder import HermesHFConfig, HermesHFEncoder
from hermes_action_transducer.models import RobotObservation
from hermes_action_transducer.profiles import get_profile


def _parse_proprio(raw: str) -> list[float]:
    if not raw.strip():
        return []
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_layer_indices(raw: str) -> tuple[int, ...]:
    if not raw.strip():
        return ()
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract Hermes hidden-state summary via Hugging Face")
    ap.add_argument("--task", required=True)
    ap.add_argument("--state-text", default="")
    ap.add_argument("--image-path", default=None)
    ap.add_argument("--proprio", default="")
    ap.add_argument("--robot-profile", choices=["arm", "go2", "g1"], default="arm")
    ap.add_argument("--model-id", default="NousResearch/Hermes-4.3-36B")
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--torch-dtype", default="auto")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--layer-index", type=int, default=-1)
    ap.add_argument("--pool-strategy", choices=["mean", "last_token"], default="mean")
    ap.add_argument("--rich-projection-dim", type=int, default=128)
    ap.add_argument("--layer-projection-dim", type=int, default=64)
    ap.add_argument("--additional-layer-indices", default="-4,-8")
    ap.add_argument("--attn-implementation", default=None)
    args = ap.parse_args()

    profile = get_profile(args.robot_profile)
    observation = RobotObservation(
        task=args.task,
        state_text=args.state_text,
        image_path=args.image_path,
        proprio=_parse_proprio(args.proprio),
    )
    encoder = HermesHFEncoder(
        HermesHFConfig(
            model_id=args.model_id,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            max_length=args.max_length,
            layer_index=args.layer_index,
            pool_strategy=args.pool_strategy,
            rich_projection_dim=args.rich_projection_dim,
            layer_projection_dim=args.layer_projection_dim,
            additional_layer_indices=_parse_layer_indices(args.additional_layer_indices),
            attn_implementation=args.attn_implementation,
        )
    )
    state = encoder.encode(observation, profile)
    print(json.dumps(state.to_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
