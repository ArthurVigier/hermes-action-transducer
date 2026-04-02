from __future__ import annotations

from hermes_action_transducer.models import RobotObservation
from hermes_action_transducer.pipeline import ActionPipeline


def test_arm_profile_prefers_manipulation_tools():
    pipeline = ActionPipeline()
    result = pipeline.run(
        RobotObservation(task="Pick up the mug and place it on the coaster"),
        robot_profile="arm",
    )
    assert result.action_ir.mode == "manipulation"
    assert result.compiled_action.name == "pick"
    assert result.profile.name == "arm"


def test_go2_profile_prefers_navigation_tools():
    pipeline = ActionPipeline()
    result = pipeline.run(
        RobotObservation(task="Go to the next room and inspect the door"),
        robot_profile="go2",
    )
    assert result.profile.name == "go2"
    assert result.action_ir.mode == "inspection" or result.action_ir.mode == "navigation"
    assert result.compiled_action.name in {"goto", "look_at", "relative_move"}


def test_halt_request_compiles_to_stop():
    pipeline = ActionPipeline()
    result = pipeline.run(
        RobotObservation(task="Stop immediately and freeze"),
        robot_profile="g1",
    )
    assert result.action_ir.mode == "halt"
    assert result.compiled_action.name == "stop"
