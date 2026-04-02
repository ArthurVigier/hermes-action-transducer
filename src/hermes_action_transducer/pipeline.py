from __future__ import annotations

from hermes_action_transducer.compiler import DimosMCPCompiler
from hermes_action_transducer.encoder import SimpleHermesEncoder
from hermes_action_transducer.models import PipelineResult, RobotObservation
from hermes_action_transducer.profiles import DEFAULT_PROFILE_NAME, get_profile
from hermes_action_transducer.transducer import ProfileAwareActionTransducer


class ActionPipeline:
    def __init__(
        self,
        *,
        encoder=None,
        transducer=None,
        compiler=None,
    ) -> None:
        self.encoder = encoder or SimpleHermesEncoder()
        self.transducer = transducer or ProfileAwareActionTransducer()
        self.compiler = compiler or DimosMCPCompiler()

    def run(self, observation: RobotObservation, *, robot_profile: str = DEFAULT_PROFILE_NAME) -> PipelineResult:
        profile = get_profile(robot_profile)
        hermes_state = self.encoder.encode(observation, profile)
        action_ir = self.transducer.predict(hermes_state, observation, profile)
        compiled_action = self.compiler.compile(action_ir, observation, profile)
        return PipelineResult(
            profile=profile,
            hermes_state=hermes_state,
            action_ir=action_ir,
            compiled_action=compiled_action,
        )
