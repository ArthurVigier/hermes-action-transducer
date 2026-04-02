from __future__ import annotations

from hermes_action_transducer.models import (
    ActionIR,
    ActionTransducer,
    HermesState,
    RobotObservation,
    RobotProfileSpec,
    TargetRef,
)


class ProfileAwareActionTransducer(ActionTransducer):
    """
    Heuristic transducer that preserves an open-ended Hermes layer while producing a compact ActionIR.
    """

    def predict(
        self,
        hermes_state: HermesState,
        observation: RobotObservation,
        profile: RobotProfileSpec,
    ) -> ActionIR:
        task_lower = observation.task.lower()
        mode = _infer_mode(task_lower, profile)
        targets = _infer_targets(task_lower)
        constraints = {
            **profile.preferred_constraints,
            **_infer_constraints(task_lower),
        }
        skill_prior = _skill_prior_for(mode, profile)
        confidence = _infer_confidence(task_lower, profile)
        subgoal = _infer_subgoal(observation.task, mode)

        return ActionIR(
            mode=mode,
            subgoal=subgoal,
            targets=targets,
            constraints=constraints,
            affordances=_infer_affordances(mode, profile),
            skill_prior=skill_prior,
            motion_latent=hermes_state.intent_vector[:6],
            safety_latent=hermes_state.thought_vector[:6],
            horizon=_infer_horizon(task_lower),
            confidence=confidence,
            metadata={
                "profile": profile.name,
                "source": "profile-aware-transducer",
                "hermes_summary": hermes_state.summary_text,
            },
        )


def _infer_mode(task_lower: str, profile: RobotProfileSpec) -> str:
    if any(word in task_lower for word in ["stop", "halt", "freeze"]):
        return "halt"
    if any(word in task_lower for word in ["pick", "place", "grasp", "put", "lift"]):
        return "manipulation" if "manipulation" in profile.supported_modes else profile.default_mode
    if any(word in task_lower for word in ["follow", "track", "keep up"]):
        return "tracking" if "tracking" in profile.supported_modes else profile.default_mode
    if any(word in task_lower for word in ["inspect", "look", "observe", "check"]):
        return "inspection" if "inspection" in profile.supported_modes else profile.default_mode
    if any(word in task_lower for word in ["go", "move", "navigate", "approach", "room"]):
        return "navigation" if "navigation" in profile.supported_modes else profile.default_mode
    return profile.default_mode


def _infer_targets(task_lower: str) -> list[TargetRef]:
    targets: list[TargetRef] = []
    for word in ["mug", "cup", "coaster", "door", "room", "table", "object", "person"]:
        if word in task_lower:
            kind = "location" if word in {"room", "table", "door"} else "object"
            targets.append(TargetRef(kind=kind, name=word))
    if not targets:
        targets.append(TargetRef(kind="task_anchor", name="primary_target"))
    return targets


def _infer_constraints(task_lower: str) -> dict[str, object]:
    constraints: dict[str, object] = {}
    if any(word in task_lower for word in ["careful", "carefully", "safe", "gently", "slow"]):
        constraints["speed"] = "slow"
        constraints["caution"] = 0.95
    if any(word in task_lower for word in ["quick", "quickly", "fast", "urgent"]):
        constraints["speed"] = "fast"
        constraints["caution"] = 0.45
    if "fragile" in task_lower:
        constraints["force_limit"] = 0.2
    return constraints


def _skill_prior_for(mode: str, profile: RobotProfileSpec) -> list[str]:
    mode_to_skills = {
        "halt": ["stop"],
        "navigation": ["goto", "relative_move", "turn", "stop"],
        "tracking": ["follow_object", "look_at", "stop"],
        "inspection": ["look_at", "goto", "stop"],
        "manipulation": ["pick", "place", "reach", "stop"],
    }
    candidates = mode_to_skills.get(mode, [profile.runtime_tools[0], "stop"])
    return [skill for skill in candidates if skill in profile.runtime_tools]


def _infer_affordances(mode: str, profile: RobotProfileSpec) -> dict[str, float]:
    base = {
        "reachable": 0.75,
        "safe": round(float(profile.preferred_constraints.get("caution", 0.7)), 2),
    }
    if mode == "manipulation":
        base["graspable"] = 0.8
    if mode == "navigation":
        base["navigable"] = 0.82
    if mode == "tracking":
        base["trackable"] = 0.78
    return base


def _infer_subgoal(task: str, mode: str) -> str:
    prefix = {
        "halt": "stabilize immediately",
        "navigation": "move toward the next waypoint",
        "tracking": "maintain target lock",
        "inspection": "improve observation quality",
        "manipulation": "prepare the next manipulation step",
    }.get(mode, "advance the task safely")
    return f"{prefix}: {task}"


def _infer_horizon(task_lower: str) -> str:
    if any(word in task_lower for word in ["then", "after", "before", "sequence"]):
        return "multi_step"
    if any(word in task_lower for word in ["now", "immediately"]):
        return "immediate"
    return "short"


def _infer_confidence(task_lower: str, profile: RobotProfileSpec) -> float:
    confidence = 0.72
    if profile.name == "arm" and any(word in task_lower for word in ["pick", "place", "grasp"]):
        confidence += 0.12
    if profile.name == "go2" and any(word in task_lower for word in ["navigate", "room", "follow"]):
        confidence += 0.1
    if profile.name == "g1" and any(word in task_lower for word in ["pick", "move", "door"]):
        confidence += 0.06
    if "unknown" in task_lower:
        confidence -= 0.18
    return max(0.05, min(0.99, round(confidence, 2)))
