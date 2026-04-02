from __future__ import annotations

from hermes_action_transducer.models import (
    ActionIR,
    CompiledAction,
    EmbodimentCompiler,
    RobotObservation,
    RobotProfileSpec,
)


class DimosMCPCompiler(EmbodimentCompiler):
    """
    Compiler from ActionIR to a runtime-oriented command shape.

    The output is generic enough to inspect offline and can later be posted to MCP.
    """

    def compile(
        self,
        action_ir: ActionIR,
        observation: RobotObservation,
        profile: RobotProfileSpec,
    ) -> CompiledAction:
        tool_name = self._select_tool(action_ir, profile)
        arguments = {
            "task": observation.task,
            "subgoal": action_ir.subgoal,
            "targets": [target.to_dict() for target in action_ir.targets],
            "constraints": action_ir.constraints,
            "affordances": action_ir.affordances,
            "motion_latent": action_ir.motion_latent,
            "safety_latent": action_ir.safety_latent,
            "confidence": action_ir.confidence,
        }
        return CompiledAction(
            command_type="mcp_tool_call",
            name=tool_name,
            arguments=arguments,
            metadata={
                "profile": profile.name,
                "mode": action_ir.mode,
                "skill_prior": action_ir.skill_prior,
                "horizon": action_ir.horizon,
            },
        )

    def _select_tool(self, action_ir: ActionIR, profile: RobotProfileSpec) -> str:
        for skill in action_ir.skill_prior:
            if skill in profile.runtime_tools:
                return skill
        if action_ir.mode == "halt" and "stop" in profile.runtime_tools:
            return "stop"
        return profile.runtime_tools[0]
