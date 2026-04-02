from __future__ import annotations

import json
import math
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from hermes_action_transducer.encoder import SimpleHermesEncoder
from hermes_action_transducer.models import ActionIR, RobotObservation, TargetRef
from hermes_action_transducer.profiles import get_profile


def load_task_lookup_from_jsonl(path: str | Path) -> dict[int, str]:
    task_lookup: dict[int, str] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        task_index = payload.get("task_index")
        task_text = payload.get("task") or payload.get("task_text") or payload.get("text")
        if task_index is None or not task_text:
            continue
        task_lookup[int(task_index)] = str(task_text)
    return task_lookup


def convert_episode_rows_to_supervised_rows(
    rows: Iterable[dict[str, Any]],
    *,
    robot_profile: str,
    task_lookup: dict[int, str] | None = None,
    max_episodes: int | None = None,
    source_format: str = "auto",
) -> list[dict[str, Any]]:
    profile = get_profile(robot_profile)
    encoder = SimpleHermesEncoder()
    grouped: dict[int, list[dict[str, Any]]] = {}

    for row in rows:
        episode_index = int(row.get("episode_index", row.get("traj_idx", 0)))
        grouped.setdefault(episode_index, []).append(row)

    examples: list[dict[str, Any]] = []
    for episode_index in sorted(grouped):
        if max_episodes is not None and len(examples) >= max_episodes:
            break
        episode_rows = sorted(grouped[episode_index], key=lambda item: int(item.get("frame_index", 0)))
        first_row = episode_rows[0]
        fmt = _infer_source_format(first_row, source_format)
        task_text = _resolve_task_text(first_row, task_lookup)
        summary = _summarize_episode(episode_rows, fmt)
        observation = RobotObservation(
            task=task_text,
            state_text=summary["state_text"],
            proprio=summary["initial_state"],
            metadata={
                "episode_index": episode_index,
                "num_frames": len(episode_rows),
                "source_task_index": first_row.get("task_index"),
                "source_format": fmt,
                "action_dim": summary["action_dim"],
                "state_dim": summary["state_dim"],
                "mean_action_abs": summary["mean_action_abs"],
                "max_action_abs": summary["max_action_abs"],
            },
        )
        hermes_state = encoder.encode(observation, profile)
        action_ir = _build_action_ir_from_episode(
            task_text=task_text,
            profile_name=robot_profile,
            runtime_tools=profile.runtime_tools,
            preferred_constraints=profile.preferred_constraints,
            episode_rows=episode_rows,
            fmt=fmt,
            summary=summary,
        )
        examples.append(
            {
                "profile": robot_profile,
                "observation": observation.to_dict(),
                "hermes_state": hermes_state.to_dict(),
                "action_ir": action_ir.to_dict(),
            }
        )
    return examples


def _infer_source_format(row: dict[str, Any], source_format: str) -> str:
    if source_format != "auto":
        return source_format
    if any(key in row for key in ["language_instruction", "action_dict", "observation/cartesian_position"]):
        return "droid"
    if any(key in row for key in ["traj_idx", "state"]) or "ZibinDong/bridgedatav2" in str(row.get("_dataset_id", "")):
        return "bridge"
    return "generic"


def _resolve_task_text(row: dict[str, Any], task_lookup: dict[int, str] | None) -> str:
    for key in ["task", "language_instruction", "instruction", "text"]:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    task_index = row.get("task_index")
    if task_lookup and task_index is not None and int(task_index) in task_lookup:
        return task_lookup[int(task_index)]
    return f"episode_{row.get('episode_index', row.get('traj_idx', 0))}"


def _summarize_episode(episode_rows: list[dict[str, Any]], fmt: str) -> dict[str, Any]:
    first_row = episode_rows[0]
    states = [_extract_state_vector(row, fmt) for row in episode_rows]
    actions = [_extract_action_vector(row, fmt) for row in episode_rows]
    non_empty_states = [state for state in states if state]
    non_empty_actions = [action for action in actions if action]

    initial_state = non_empty_states[0] if non_empty_states else []
    final_state = non_empty_states[-1] if non_empty_states else []
    mean_action_abs = _mean_abs_over_vectors(non_empty_actions)
    max_action_abs = _max_abs_over_vectors(non_empty_actions)
    mean_action = _mean_vector(non_empty_actions, 6)
    action_delta = _delta_vector(non_empty_actions, 6)
    state_delta = _delta_between(initial_state, final_state, 6)
    gripper_series = [action[-1] for action in non_empty_actions if action]
    gripper_delta = (gripper_series[-1] - gripper_series[0]) if len(gripper_series) >= 2 else 0.0
    reward = _episode_reward_signal(episode_rows)
    terminal = any(bool(row.get("next.done", False) or row.get("done", False)) for row in episode_rows)

    state_text = (
        f"fmt={fmt}; frames={len(episode_rows)}; state_dim={len(initial_state)}; "
        f"action_dim={(len(non_empty_actions[0]) if non_empty_actions else 0)}; "
        f"mean_action_abs={mean_action_abs:.4f}; max_action_abs={max_action_abs:.4f}; "
        f"state_delta={_short_vector(state_delta)}; action_delta={_short_vector(action_delta)}; "
        f"gripper_delta={gripper_delta:.4f}; reward={reward:.3f}; terminal={str(terminal).lower()}"
    )
    return {
        "initial_state": initial_state,
        "final_state": final_state,
        "action_dim": len(non_empty_actions[0]) if non_empty_actions else 0,
        "state_dim": len(initial_state),
        "mean_action_abs": mean_action_abs,
        "max_action_abs": max_action_abs,
        "mean_action": mean_action,
        "action_delta": action_delta,
        "state_delta": state_delta,
        "gripper_delta": gripper_delta,
        "reward": reward,
        "terminal": terminal,
        "state_text": state_text,
    }


def _build_action_ir_from_episode(
    *,
    task_text: str,
    profile_name: str,
    runtime_tools: list[str],
    preferred_constraints: dict[str, Any],
    episode_rows: list[dict[str, Any]],
    fmt: str,
    summary: dict[str, Any],
) -> ActionIR:
    task_lower = task_text.lower()
    mode = _infer_mode(task_lower, profile_name)
    skill_prior = _infer_skill_prior(task_lower, mode, profile_name, runtime_tools, summary)
    caution = _infer_caution(task_lower, summary, preferred_constraints)
    speed = _infer_speed(summary["mean_action_abs"], task_lower, preferred_constraints)
    force_limit = 0.2 if any(word in task_lower for word in ["fragile", "careful", "gently"]) else 0.0
    confidence = _infer_confidence(summary, task_lower)
    motion_latent = _build_motion_latent(summary)
    safety_latent = _build_safety_latent(summary, caution, confidence)
    targets = _infer_targets(task_lower)
    if not targets:
        targets = [TargetRef(kind="task_anchor", name="primary_target")]
    horizon = _infer_horizon(task_lower, len(episode_rows))
    affordances = _infer_affordances(mode, profile_name, summary, caution)

    return ActionIR(
        mode=mode,
        subgoal=_infer_subgoal(task_text, mode),
        targets=targets,
        constraints={
            **preferred_constraints,
            "speed": speed,
            "caution": caution,
            "force_limit": force_limit,
            "terminal_episode": summary["terminal"],
        },
        affordances=affordances,
        skill_prior=skill_prior,
        motion_latent=motion_latent,
        safety_latent=safety_latent,
        horizon=horizon,
        confidence=confidence,
        metadata={
            "source": "specialized-converter",
            "source_format": fmt,
            "reward": summary["reward"],
            "gripper_delta": round(summary["gripper_delta"], 4),
        },
    )


def _extract_state_vector(row: dict[str, Any], fmt: str) -> list[float]:
    keys = ["observation.state", "state"]
    if fmt == "droid":
        keys = ["observation.state", "state", "observation/cartesian_position"]
    for key in keys:
        value = row.get(key)
        if isinstance(value, list):
            return [float(x) for x in value]
    return []


def _extract_action_vector(row: dict[str, Any], fmt: str) -> list[float]:
    keys = ["action", "action.original"]
    if fmt == "droid":
        keys = ["action", "action.original", "action_dict.cartesian_velocity"]
    for key in keys:
        value = row.get(key)
        if isinstance(value, list):
            return [float(x) for x in value]
    action_dict = row.get("action_dict")
    if isinstance(action_dict, dict):
        for key in ["cartesian_velocity", "cartesian_position", "joint_velocity"]:
            value = action_dict.get(key)
            if isinstance(value, list):
                return [float(x) for x in value]
    return []


def _episode_reward_signal(episode_rows: list[dict[str, Any]]) -> float:
    rewards = []
    for row in episode_rows:
        for key in ["reward", "next.reward"]:
            value = row.get(key)
            if isinstance(value, (int, float)):
                rewards.append(float(value))
    if rewards:
        return max(rewards)
    successes = [row.get("success") for row in episode_rows]
    if any(bool(value) for value in successes):
        return 1.0
    return 1.0 if any(bool(row.get("next.done", False) or row.get("done", False)) for row in episode_rows) else 0.0


def _infer_mode(task_lower: str, profile_name: str) -> str:
    if any(word in task_lower for word in ["stop", "halt", "freeze"]):
        return "halt"
    if profile_name == "arm":
        return "manipulation"
    if any(word in task_lower for word in ["follow", "track"]):
        return "tracking"
    if any(word in task_lower for word in ["inspect", "look", "observe", "check"]):
        return "inspection"
    if any(word in task_lower for word in ["pick", "place", "grasp", "put", "lift"]):
        return "manipulation"
    return "navigation"


def _infer_skill_prior(
    task_lower: str,
    mode: str,
    profile_name: str,
    runtime_tools: list[str],
    summary: dict[str, Any],
) -> list[str]:
    candidates: list[str]
    if mode == "halt":
        candidates = ["stop"]
    elif mode == "tracking":
        candidates = ["follow_object", "look_at", "stop"]
    elif mode == "inspection":
        candidates = ["look_at", "goto", "relative_move", "stop"]
    elif mode == "navigation":
        if any(word in task_lower for word in ["turn", "rotate"]):
            candidates = ["turn", "relative_move", "goto", "stop"]
        else:
            candidates = ["goto", "relative_move", "turn", "stop"]
    else:
        if any(word in task_lower for word in ["place", "put", "drop"]):
            candidates = ["place", "reach", "pick", "stop"]
        elif any(word in task_lower for word in ["pick", "grasp", "lift"]) or summary["gripper_delta"] < -0.05:
            candidates = ["pick", "reach", "place", "stop"]
        else:
            candidates = ["reach", "pick", "place", "stop"]
        if profile_name != "arm" and "pick" not in runtime_tools:
            candidates = ["relative_move", "goto", "stop"]
    return [skill for skill in candidates if skill in runtime_tools] or [runtime_tools[0]]


def _infer_caution(task_lower: str, summary: dict[str, Any], preferred_constraints: dict[str, Any]) -> float:
    caution = float(preferred_constraints.get("caution", 0.7))
    if any(word in task_lower for word in ["careful", "carefully", "safe", "gently", "fragile"]):
        caution = max(caution, 0.92)
    if summary["max_action_abs"] > 0.2:
        caution = min(0.99, caution + 0.04)
    if summary["reward"] <= 0.0 and not summary["terminal"]:
        caution = min(0.99, caution + 0.03)
    return round(caution, 4)


def _infer_speed(mean_action_abs: float, task_lower: str, preferred_constraints: dict[str, Any]) -> str:
    if any(word in task_lower for word in ["slow", "careful", "gently"]):
        return "slow"
    if any(word in task_lower for word in ["fast", "quick", "urgent"]):
        return "fast"
    if mean_action_abs < 0.02:
        return "slow"
    if mean_action_abs > 0.09:
        return "fast"
    return str(preferred_constraints.get("speed", "normal"))


def _infer_confidence(summary: dict[str, Any], task_lower: str) -> float:
    confidence = 0.55
    confidence += min(0.2, summary["reward"] * 0.2)
    confidence += 0.08 if summary["terminal"] else 0.0
    confidence += 0.06 if "careful" not in task_lower else 0.03
    confidence -= 0.08 if "unknown" in task_lower else 0.0
    confidence -= 0.05 if summary["max_action_abs"] > 0.3 else 0.0
    return max(0.05, min(0.99, round(confidence, 4)))


def _build_motion_latent(summary: dict[str, Any]) -> list[float]:
    return _pad_to_six(
        summary["mean_action"][:3]
        + summary["action_delta"][:3]
    )


def _build_safety_latent(summary: dict[str, Any], caution: float, confidence: float) -> list[float]:
    return _pad_to_six(
        [
            caution,
            confidence,
            summary["mean_action_abs"],
            summary["max_action_abs"],
            abs(summary["gripper_delta"]),
            float(summary["reward"]),
        ]
    )


def _infer_targets(task_lower: str) -> list[TargetRef]:
    targets: list[TargetRef] = []
    for word in ["mug", "cup", "coaster", "door", "room", "table", "object", "person"]:
        if word in task_lower:
            kind = "location" if word in {"room", "table", "door"} else "object"
            targets.append(TargetRef(kind=kind, name=word))
    return targets


def _infer_horizon(task_lower: str, frame_count: int) -> str:
    if any(word in task_lower for word in ["immediately", "now", "stop"]):
        return "immediate"
    if any(word in task_lower for word in ["then", "after", "before", "and"]) or frame_count > 40:
        return "multi_step"
    return "short"


def _infer_affordances(mode: str, profile_name: str, summary: dict[str, Any], caution: float) -> dict[str, float]:
    affordances = {
        "reachable": round(max(0.2, 1.0 - min(0.6, summary["mean_action_abs"] * 3.0)), 4),
        "safe": round(1.0 - max(0.0, caution - 0.1), 4),
    }
    if mode == "manipulation":
        affordances["graspable"] = round(max(0.2, 0.8 - abs(summary["gripper_delta"]) * 0.2), 4)
    if mode == "navigation":
        affordances["navigable"] = round(max(0.2, 0.85 - summary["max_action_abs"] * 0.3), 4)
    if mode == "tracking":
        affordances["trackable"] = 0.78
    if profile_name == "g1":
        affordances["whole_body_ready"] = 0.72
    return affordances


def _infer_subgoal(task_text: str, mode: str) -> str:
    prefix = {
        "halt": "stabilize immediately",
        "navigation": "advance toward the target",
        "tracking": "maintain target lock",
        "inspection": "improve observation quality",
        "manipulation": "execute the next manipulation phase",
    }.get(mode, "advance safely")
    return f"{prefix}: {task_text}"


def _mean_abs_over_vectors(vectors: list[list[float]]) -> float:
    if not vectors:
        return 0.0
    flat = [abs(value) for vector in vectors for value in vector]
    return float(mean(flat)) if flat else 0.0


def _max_abs_over_vectors(vectors: list[list[float]]) -> float:
    if not vectors:
        return 0.0
    return max(abs(value) for vector in vectors for value in vector)


def _mean_vector(vectors: list[list[float]], max_dim: int) -> list[float]:
    if not vectors:
        return [0.0] * max_dim
    dim = min(max_dim, max(len(vector) for vector in vectors))
    out = []
    for idx in range(dim):
        values = [vector[idx] for vector in vectors if len(vector) > idx]
        out.append(float(mean(values)) if values else 0.0)
    return _pad_to_six(out)


def _delta_vector(vectors: list[list[float]], max_dim: int) -> list[float]:
    if len(vectors) < 2:
        return [0.0] * max_dim
    first = vectors[0]
    last = vectors[-1]
    dim = min(max_dim, len(first), len(last))
    out = [last[idx] - first[idx] for idx in range(dim)]
    return _pad_to_six(out)


def _delta_between(first: list[float], last: list[float], max_dim: int) -> list[float]:
    if not first or not last:
        return [0.0] * max_dim
    dim = min(max_dim, len(first), len(last))
    return _pad_to_six([last[idx] - first[idx] for idx in range(dim)])


def _pad_to_six(values: list[float]) -> list[float]:
    clipped = [round(float(value), 4) for value in values[:6]]
    return clipped + [0.0] * max(0, 6 - len(clipped))


def _short_vector(values: list[float], *, max_items: int = 4) -> str:
    if not values:
        return "[]"
    clipped = values[:max_items]
    rendered = ", ".join(f"{value:.3f}" for value in clipped)
    if len(values) > max_items:
        rendered += ", ..."
    return f"[{rendered}]"
