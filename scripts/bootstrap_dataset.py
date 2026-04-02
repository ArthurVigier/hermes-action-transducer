#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hermes_action_transducer.encoder import SimpleHermesEncoder
from hermes_action_transducer.data_io import save_jsonl_dataset
from hermes_action_transducer.pipeline import ActionPipeline
from hermes_action_transducer.models import RobotObservation
from hermes_action_transducer.profiles import PROFILE_REGISTRY


SEED_TASKS = [
    ("arm", "Pick up the mug and place it on the coaster", "mug left of gripper"),
    ("arm", "Carefully place the cup on the table", "cup already grasped"),
    ("arm", "Stop immediately", "arm moving toward table"),
    ("go2", "Go to the next room", "hallway clear"),
    ("go2", "Follow the person carefully", "person ahead at two meters"),
    ("go2", "Inspect the door", "door closed"),
    ("g1", "Pick up the object and move to the table", "object on floor"),
    ("g1", "Turn toward the door", "door on the right"),
    ("g1", "Stop and freeze", "mid-step"),
]


def main() -> int:
    out_path = ROOT / "data" / "bootstrap_train.jsonl"
    pipeline = ActionPipeline()
    encoder = SimpleHermesEncoder()
    rows = []
    for profile_name, task, state_text in SEED_TASKS:
        observation = RobotObservation(task=task, state_text=state_text)
        result = pipeline.run(observation, robot_profile=profile_name)
        hermes_state = encoder.encode(observation, PROFILE_REGISTRY[profile_name])
        rows.append(
            {
                "profile": profile_name,
                "observation": observation.to_dict(),
                "hermes_state": hermes_state.to_dict(),
                "action_ir": result.action_ir.to_dict(),
            }
        )

    save_jsonl_dataset(out_path, rows)
    print(json.dumps({"saved": str(out_path), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
