from __future__ import annotations

import json
from pathlib import Path

from hermes_action_transducer.conversion import (
    convert_episode_rows_to_supervised_rows,
    load_task_lookup_from_jsonl,
)


def test_conversion_groups_rows_by_episode():
    rows = [
        {
            "episode_index": 0,
            "frame_index": 1,
            "task_index": 3,
            "observation.state": [0.1] * 7,
            "action": [0.0] * 7,
        },
        {
            "episode_index": 0,
            "frame_index": 0,
            "task_index": 3,
            "observation.state": [0.2] * 7,
            "action": [1.0] * 7,
        },
        {
            "episode_index": 1,
            "frame_index": 0,
            "task_index": 9,
            "observation.state": [0.3] * 7,
            "action": [0.1] * 7,
        },
    ]
    examples = convert_episode_rows_to_supervised_rows(
        rows,
        robot_profile="arm",
        task_lookup={3: "Pick up the mug", 9: "Place the cup on the table"},
    )
    assert len(examples) == 2
    assert examples[0]["observation"]["task"] == "Pick up the mug"
    assert examples[0]["observation"]["metadata"]["num_frames"] == 2
    assert examples[1]["action_ir"]["mode"] == "manipulation"
    assert examples[0]["action_ir"]["metadata"]["source"] == "specialized-converter"


def test_load_task_lookup_from_jsonl(tmp_path: Path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"task_index": 1, "task": "Go to the room"}),
                json.dumps({"task_index": 2, "task_text": "Inspect the door"}),
            ]
        ),
        encoding="utf-8",
    )
    lookup = load_task_lookup_from_jsonl(path)
    assert lookup[1] == "Go to the room"
    assert lookup[2] == "Inspect the door"


def test_droid_conversion_uses_language_and_reward_signals():
    rows = [
        {
            "episode_index": 0,
            "frame_index": 0,
            "language_instruction": "Pick up the mug carefully",
            "observation.state": [0.0] * 7,
            "action.original": [0.02, 0.01, -0.03, 0.0, 0.0, 0.0, -1.0],
            "next.reward": 1.0,
            "next.done": True,
        },
        {
            "episode_index": 0,
            "frame_index": 1,
            "language_instruction": "Pick up the mug carefully",
            "observation.state": [0.1] * 7,
            "action.original": [0.01, 0.01, -0.02, 0.0, 0.0, 0.0, -0.5],
            "next.reward": 1.0,
            "next.done": True,
        },
    ]
    examples = convert_episode_rows_to_supervised_rows(
        rows,
        robot_profile="arm",
        source_format="droid",
    )
    example = examples[0]
    assert example["observation"]["task"] == "Pick up the mug carefully"
    assert example["action_ir"]["skill_prior"][0] == "pick"
    assert example["action_ir"]["constraints"]["caution"] >= 0.9
    assert example["action_ir"]["confidence"] >= 0.8


def test_bridge_conversion_uses_task_lookup_and_navigation_bias():
    rows = [
        {
            "traj_idx": 14,
            "episode_index": 3,
            "frame_index": 0,
            "task_index": 7,
            "state": [0.1] * 7,
            "action": [0.12, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0],
        },
        {
            "traj_idx": 14,
            "episode_index": 3,
            "frame_index": 1,
            "task_index": 7,
            "state": [0.15] * 7,
            "action": [0.1, 0.01, 0.0, 0.02, 0.0, 0.0, 0.0],
        },
    ]
    examples = convert_episode_rows_to_supervised_rows(
        rows,
        robot_profile="go2",
        task_lookup={7: "Go to the next room"},
        source_format="bridge",
    )
    example = examples[0]
    assert example["observation"]["task"] == "Go to the next room"
    assert example["action_ir"]["mode"] == "navigation"
    assert example["action_ir"]["skill_prior"][0] in {"goto", "relative_move"}
