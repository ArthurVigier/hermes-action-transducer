from __future__ import annotations

from hermes_action_transducer.models import RobotProfileSpec


ARM_PROFILE = RobotProfileSpec(
    name="arm",
    description="Tabletop manipulation profile for robot arms.",
    default_mode="manipulation",
    supported_modes=["manipulation", "inspection", "halt"],
    runtime_tools=["pick", "place", "reach", "look_at", "stop"],
    preferred_constraints={"speed": "slow", "caution": 0.8},
)

GO2_PROFILE = RobotProfileSpec(
    name="go2",
    description="Mobile quadruped profile focused on navigation and inspection.",
    default_mode="navigation",
    supported_modes=["navigation", "inspection", "halt", "tracking"],
    runtime_tools=["relative_move", "turn", "goto", "look_at", "follow_object", "stop"],
    preferred_constraints={"speed": "normal", "caution": 0.7},
)

G1_PROFILE = RobotProfileSpec(
    name="g1",
    description="Humanoid profile mixing navigation and manipulation.",
    default_mode="navigation",
    supported_modes=["navigation", "manipulation", "inspection", "halt", "tracking"],
    runtime_tools=["relative_move", "turn", "goto", "pick", "place", "look_at", "stop"],
    preferred_constraints={"speed": "normal", "caution": 0.85},
)


PROFILE_REGISTRY: dict[str, RobotProfileSpec] = {
    ARM_PROFILE.name: ARM_PROFILE,
    GO2_PROFILE.name: GO2_PROFILE,
    G1_PROFILE.name: G1_PROFILE,
}

DEFAULT_PROFILE_NAME = ARM_PROFILE.name


def get_profile(name: str) -> RobotProfileSpec:
    try:
        return PROFILE_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown robot profile: {name}") from exc
