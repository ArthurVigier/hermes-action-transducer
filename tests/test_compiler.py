from __future__ import annotations

from hermes_action_transducer.compiler import DimosMCPCompiler
from hermes_action_transducer.models import ActionIR, RobotObservation, TargetRef
from hermes_action_transducer.profiles import get_profile


def test_compiler_selects_first_supported_skill():
    compiler = DimosMCPCompiler()
    profile = get_profile("go2")
    action_ir = ActionIR(
        mode="navigation",
        subgoal="move toward the room",
        targets=[TargetRef(kind="location", name="room")],
        skill_prior=["goto", "relative_move"],
        confidence=0.8,
    )
    compiled = compiler.compile(action_ir, RobotObservation(task="Go to room"), profile)
    assert compiled.command_type == "mcp_tool_call"
    assert compiled.name == "goto"
    assert compiled.arguments["targets"][0]["name"] == "room"
