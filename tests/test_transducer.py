from __future__ import annotations

from hermes_action_transducer.encoder import SimpleHermesEncoder
from hermes_action_transducer.models import RobotObservation
from hermes_action_transducer.profiles import get_profile
from hermes_action_transducer.transducer import ProfileAwareActionTransducer


def test_transducer_preserves_rich_action_ir():
    profile = get_profile("arm")
    observation = RobotObservation(
        task="Carefully pick the mug and place it on the coaster",
        state_text="mug is left of the gripper",
    )
    encoder = SimpleHermesEncoder()
    state = encoder.encode(observation, profile)
    action_ir = ProfileAwareActionTransducer().predict(state, observation, profile)

    assert action_ir.mode == "manipulation"
    assert action_ir.constraints["speed"] == "slow"
    assert action_ir.constraints["caution"] >= 0.9
    assert "pick" in action_ir.skill_prior
    assert len(action_ir.motion_latent) == 6
    assert len(action_ir.targets) >= 1
