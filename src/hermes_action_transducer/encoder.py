from __future__ import annotations

from hermes_action_transducer.models import HermesEncoder, HermesState, RobotObservation, RobotProfileSpec


class SimpleHermesEncoder(HermesEncoder):
    """
    Placeholder encoder.

    This keeps the architecture explicit now so we can later swap in:
    - hidden-state extraction from Hermes
    - pooled activations
    - a learned projection head
    """

    def encode(self, observation: RobotObservation, profile: RobotProfileSpec) -> HermesState:
        task = observation.task.strip()
        state = observation.state_text.strip() or "no state"
        summary = (
            f"profile={profile.name}; task={task}; state={state}; "
            f"tools={','.join(profile.runtime_tools[:4])}"
        )
        thought_vector = _string_to_vector(f"{task}|{state}|{profile.name}", size=8)
        intent_vector = _string_to_vector(task, size=8)
        return HermesState(
            summary_text=summary,
            thought_vector=thought_vector,
            intent_vector=intent_vector,
            metadata={
                "profile": profile.name,
                "task_length": len(task.split()),
                "has_image": observation.image_path is not None,
            },
        )


def _string_to_vector(raw: str, *, size: int) -> list[float]:
    values = [0.0] * size
    if not raw:
        return values
    for idx, ch in enumerate(raw.encode("utf-8")):
        values[idx % size] += ((ch % 31) / 30.0)
    return [round(value / max(1, len(raw)), 4) for value in values]
