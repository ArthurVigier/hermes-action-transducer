from __future__ import annotations

import json
from pathlib import Path

from hermes_action_transducer.features import FeatureConfig
from hermes_action_transducer.models import ActionIR
from hermes_action_transducer.plan_stack import (
    PlanStackTrainingConfig,
    build_control_target,
    discover_plan_codebook,
    derive_plan_code,
    train_plan_stack,
)


def test_derive_plan_code_uses_mode_tool_and_horizon():
    action_ir = ActionIR(
        mode="manipulation",
        subgoal="pick up mug",
        skill_prior=["pick", "reach"],
        motion_latent=[0.0] * 6,
        safety_latent=[0.0] * 6,
        horizon="short",
    )
    assert derive_plan_code(action_ir) == "manipulation::pick::short"


def test_build_control_target_has_expected_shape():
    action_ir = ActionIR(
        mode="manipulation",
        subgoal="pick up mug",
        skill_prior=["pick"],
        constraints={"speed": "fast", "caution": 0.25, "force_limit": 0.1},
        motion_latent=[0.1] * 6,
        safety_latent=[0.2] * 6,
        confidence=0.9,
        horizon="short",
    )
    target = build_control_target(action_ir)
    assert len(target) == 16
    assert target[0] == 1.0
    assert target[1] == 0.9


def test_discover_plan_codebook_learns_cluster_assignments():
    action_irs = [
        ActionIR(
            mode="manipulation",
            subgoal="pick up mug",
            skill_prior=["pick"],
            motion_latent=[0.1] * 6,
            safety_latent=[0.0] * 6,
            horizon="short",
        ),
        ActionIR(
            mode="manipulation",
            subgoal="place mug down",
            skill_prior=["place"],
            motion_latent=[0.9] * 6,
            safety_latent=[0.2] * 6,
            horizon="short",
        ),
    ]
    codebook = discover_plan_codebook(action_irs, max_codes=2, kmeans_iterations=4)
    assert codebook.plan_vocab == ["plan_00", "plan_01"]
    assert len(codebook.assignment_indices) == 2
    assert len(codebook.representative_codes) == 2
    assert set(codebook.representative_codes) == {"manipulation::pick::short", "manipulation::place::short"}


def test_train_plan_stack_smoke(tmp_path: Path):
    dataset_path = tmp_path / "plan_stack.jsonl"
    checkpoint_dir = tmp_path / "ckpts"
    rows = [
        _example_row("pick up mug", "manipulation", "pick", "short", [0.2] * 8, [0.1] * 6),
        _example_row("place mug down", "manipulation", "place", "short", [0.3] * 8, [0.2] * 6),
        _example_row("go to docking station", "navigation", "goto", "multi_step", [0.4] * 8, [0.3] * 6),
    ]
    dataset_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    result = train_plan_stack(
        dataset_path,
        checkpoint_dir,
        config=PlanStackTrainingConfig(
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            hidden_dim=32,
            device="cpu",
            feature_config=FeatureConfig(mode="compact"),
            max_plan_codes=3,
            kmeans_iterations=4,
        ),
    )

    assert result["plan_vocab"] == ["plan_00", "plan_01", "plan_02"]
    assert len(result["plan_codebook"]["assignment_indices"]) == 3
    assert set(result["plan_codebook"]["representative_codes"]) == {
        "manipulation::pick::short",
        "manipulation::place::short",
        "navigation::goto::multi_step",
    }
    assert (checkpoint_dir / "plan_predictor.pt").exists()
    assert (checkpoint_dir / "control_policy.pt").exists()
    assert "plan_acc" in result["plan_metrics"]["eval"]
    assert "control_mse" in result["control_metrics"]["eval"]


def _example_row(
    task: str,
    mode: str,
    tool: str,
    horizon: str,
    state_vec: list[float],
    latent_vec: list[float],
) -> dict:
    return {
        "profile": "arm",
        "observation": {
            "task": task,
            "state_text": f"state for {task}",
            "proprio": state_vec,
            "metadata": {},
        },
        "hermes_state": {
            "summary_text": f"hermes summary for {task}",
            "thought_vector": [0.1] * 8,
            "intent_vector": [0.2] * 8,
            "hidden_projection": [0.3] * 16,
            "layer_projections": {},
            "metadata": {},
        },
        "action_ir": {
            "mode": mode,
            "subgoal": task,
            "targets": [],
            "constraints": {"speed": "normal", "caution": 0.2, "force_limit": 0.1},
            "affordances": {},
            "skill_prior": [tool],
            "motion_latent": latent_vec,
            "safety_latent": [value / 2.0 for value in latent_vec],
            "horizon": horizon,
            "confidence": 0.8,
            "metadata": {},
        },
    }
