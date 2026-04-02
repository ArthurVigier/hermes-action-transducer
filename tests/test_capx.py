from __future__ import annotations

from pathlib import Path

from hermes_action_transducer.capx import (
    build_capx_launch_command,
    discover_capx_suite_configs,
    infer_capx_tier,
    parse_capx_summary_text,
    resolve_capx_tiers,
)


def test_resolve_capx_complete_tiers():
    assert resolve_capx_tiers("complete") == ["S1", "S2", "S3", "S4", "M1", "M2", "M3", "M4"]


def test_infer_capx_tier_from_official_like_filenames():
    assert infer_capx_tier("franka_robosuite_cube_stack_privileged.yaml") == "S1"
    assert infer_capx_tier("franka_robosuite_cube_stack.yaml") == "S2"
    assert infer_capx_tier("franka_robosuite_cube_stack_reduced_api.yaml") == "S3"
    assert infer_capx_tier("franka_robosuite_cube_stack_reduced_api_exampleless.yaml") == "S4"
    assert infer_capx_tier("franka_robosuite_cube_stack_multiturn.yaml") == "M1"
    assert infer_capx_tier("franka_robosuite_cube_stack_multiturn_vf.yaml") == "M2"
    assert infer_capx_tier("franka_robosuite_cube_stack_multiturn_vdm.yaml") == "M3"
    assert infer_capx_tier("franka_robosuite_cube_stack_multiturn_vdm_reduced_api.yaml") == "M4"
    assert infer_capx_tier("franka_robosuite_cube_stack_multiturn_vdm_reduced_api_skill_lib.yaml") is None


def test_discover_capx_suite_configs(tmp_path: Path):
    suite_dir = tmp_path / "env_configs" / "cube_stack"
    suite_dir.mkdir(parents=True)
    for name in [
        "franka_robosuite_cube_stack_privileged.yaml",
        "franka_robosuite_cube_stack.yaml",
        "franka_robosuite_cube_stack_reduced_api.yaml",
        "franka_robosuite_cube_stack_reduced_api_exampleless.yaml",
        "franka_robosuite_cube_stack_multiturn.yaml",
        "franka_robosuite_cube_stack_multiturn_vf.yaml",
        "franka_robosuite_cube_stack_multiturn_vdm.yaml",
        "franka_robosuite_cube_stack_multiturn_vdm_reduced_api.yaml",
    ]:
        (suite_dir / name).write_text("trials: 100\n", encoding="utf-8")

    discovered = discover_capx_suite_configs(tmp_path, "cube_stack")
    assert set(discovered) == {"S1", "S2", "S3", "S4", "M1", "M2", "M3", "M4"}
    assert discovered["M3"].interaction_mode == "multi_turn"


def test_parse_capx_summary_text():
    text = """
Summary Statistics:
Model: NousResearch/Hermes-4.3-36B
Total number of trials: 100
Code generation success rate / Average reward / Task completed:
0.810/0.612/73
Average code blocks: 1.500
Average regenerations: 0.250
Average finishes: 0.730
Elapsed time: 123.45 seconds
"""
    parsed = parse_capx_summary_text(text)
    assert parsed is not None
    assert parsed["success_rate"] == 0.81
    assert parsed["task_completed_count"] == 73
    assert parsed["task_completed_rate"] == 0.73
    assert parsed["elapsed_seconds"] == 123.45


def test_build_capx_launch_command():
    capx_root = _build_fake_capx_root(Path("/tmp"))
    command = build_capx_launch_command(
        capx_root=capx_root,
        spec=discover_capx_suite_configs(
            capx_root,
            "cube_stack",
        )["S2"],
        model="NousResearch/Hermes-4.3-36B",
        server_url="http://127.0.0.1:8110/chat/completions",
        total_trials=20,
        num_workers=4,
    )
    assert command[:5] == ["uv", "run", "--no-sync", "--active", "capx/envs/launch.py"]
    assert "--total-trials" in command
    assert "--num-workers" in command
    assert command[command.index("--config-path") + 1].endswith("env_configs/cube_stack/franka_robosuite_cube_stack.yaml")


def _build_fake_capx_root(base: Path) -> Path:
    capx_root = base / "capx_test_root"
    suite_dir = capx_root / "env_configs" / "cube_stack"
    suite_dir.mkdir(parents=True, exist_ok=True)
    (suite_dir / "franka_robosuite_cube_stack.yaml").write_text("trials: 100\n", encoding="utf-8")
    return capx_root
