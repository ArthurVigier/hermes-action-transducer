from __future__ import annotations

from hermes_action_transducer.constants import MODE_VOCAB
from hermes_action_transducer.models import ActionIR, HermesState, RobotObservation, RobotProfileSpec


def build_feature_vector(
    hermes_state: HermesState,
    observation: RobotObservation,
    profile: RobotProfileSpec,
) -> list[float]:
    task_vec = _string_to_vector(observation.task, size=8)
    state_vec = _string_to_vector(observation.state_text, size=8)
    profile_vec = [1.0 if profile.default_mode == mode else 0.0 for mode in MODE_VOCAB]
    proprio = observation.proprio[:8] + [0.0] * max(0, 8 - len(observation.proprio[:8]))
    return (
        task_vec
        + state_vec
        + hermes_state.thought_vector[:8]
        + [0.0] * max(0, 8 - len(hermes_state.thought_vector[:8]))
        + hermes_state.intent_vector[:8]
        + [0.0] * max(0, 8 - len(hermes_state.intent_vector[:8]))
        + profile_vec
        + proprio[:8]
    )


def action_ir_target_summary(action_ir: ActionIR) -> dict[str, object]:
    skill = action_ir.skill_prior[0] if action_ir.skill_prior else "stop"
    speed = str(action_ir.constraints.get("speed", "normal"))
    caution = float(action_ir.constraints.get("caution", 0.7))
    force_limit = float(action_ir.constraints.get("force_limit", 0.0))
    return {
        "mode": action_ir.mode,
        "tool": skill,
        "horizon": action_ir.horizon,
        "speed": speed,
        "confidence": float(action_ir.confidence),
        "caution": caution,
        "force_limit": force_limit,
        "motion_latent": action_ir.motion_latent[:6] + [0.0] * max(0, 6 - len(action_ir.motion_latent[:6])),
        "safety_latent": action_ir.safety_latent[:6] + [0.0] * max(0, 6 - len(action_ir.safety_latent[:6])),
    }


def _string_to_vector(raw: str, *, size: int) -> list[float]:
    values = [0.0] * size
    if not raw:
        return values
    encoded = raw.encode("utf-8")
    for idx, ch in enumerate(encoded):
        values[idx % size] += ((ch % 29) / 28.0)
    scale = max(1, len(encoded))
    return [round(value / scale, 5) for value in values]
