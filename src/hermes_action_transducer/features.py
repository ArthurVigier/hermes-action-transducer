from __future__ import annotations

from dataclasses import asdict, dataclass

from hermes_action_transducer.constants import (
    BASE_FEATURE_DIM,
    LAYER_SUMMARY_DIM,
    MAX_LAYER_PROJECTIONS,
    MODE_VOCAB,
    OBSERVATION_FEATURE_DIM,
    PER_LAYER_PROJECTION_DIM,
    RICH_PROJECTION_DIM,
)
from hermes_action_transducer.models import ActionIR, HermesState, RobotObservation, RobotProfileSpec


@dataclass(frozen=True)
class FeatureConfig:
    mode: str = "rich"
    rich_projection_dim: int = RICH_PROJECTION_DIM
    layer_summary_dim: int = LAYER_SUMMARY_DIM
    per_layer_projection_dim: int = PER_LAYER_PROJECTION_DIM
    max_layer_projections: int = MAX_LAYER_PROJECTIONS

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def build_feature_vector(
    hermes_state: HermesState,
    observation: RobotObservation,
    profile: RobotProfileSpec,
    config: FeatureConfig | None = None,
) -> list[float]:
    config = config or FeatureConfig()
    task_vec = _string_to_vector(observation.task, size=8)
    state_vec = _string_to_vector(observation.state_text, size=8)
    profile_vec = [1.0 if profile.default_mode == mode else 0.0 for mode in MODE_VOCAB]
    proprio = observation.proprio[:8] + [0.0] * max(0, 8 - len(observation.proprio[:8]))
    observation_only = (
        task_vec
        + state_vec
        + profile_vec
        + proprio[:8]
    )
    compact = (
        observation_only
        + _fit_vector(hermes_state.thought_vector, size=8)
        + _fit_vector(hermes_state.intent_vector, size=8)
    )

    if config.mode == "vanilla":
        return observation_only

    if config.mode == "compact":
        return compact

    hidden_projection = _fit_vector(hermes_state.hidden_projection, size=config.rich_projection_dim)
    if config.mode == "rich":
        layer_summary = _aggregate_layer_projections(hermes_state, size=config.layer_summary_dim)
        return compact + hidden_projection + layer_summary

    if config.mode == "per_layer":
        per_layer = _flatten_layer_projections(
            hermes_state,
            projection_dim=config.per_layer_projection_dim,
            max_layers=config.max_layer_projections,
        )
        return compact + hidden_projection + per_layer

    if config.mode == "full":
        layer_summary = _aggregate_layer_projections(hermes_state, size=config.layer_summary_dim)
        per_layer = _flatten_layer_projections(
            hermes_state,
            projection_dim=config.per_layer_projection_dim,
            max_layers=config.max_layer_projections,
        )
        return compact + hidden_projection + layer_summary + per_layer

    raise ValueError(f"Unknown feature mode: {config.mode}")


def get_feature_dim(config: FeatureConfig | None = None) -> int:
    config = config or FeatureConfig()
    if config.mode == "vanilla":
        return OBSERVATION_FEATURE_DIM
    if config.mode == "compact":
        return BASE_FEATURE_DIM
    if config.mode == "rich":
        return BASE_FEATURE_DIM + config.rich_projection_dim + config.layer_summary_dim
    if config.mode == "per_layer":
        return BASE_FEATURE_DIM + config.rich_projection_dim + (
            config.per_layer_projection_dim * config.max_layer_projections
        )
    if config.mode == "full":
        return (
            BASE_FEATURE_DIM
            + config.rich_projection_dim
            + config.layer_summary_dim
            + (config.per_layer_projection_dim * config.max_layer_projections)
        )
    raise ValueError(f"Unknown feature mode: {config.mode}")


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


def _aggregate_layer_projections(hermes_state: HermesState, *, size: int) -> list[float]:
    if not hermes_state.layer_projections:
        return [0.0] * size
    out = [0.0] * size
    num_layers = 0
    for values in hermes_state.layer_projections.values():
        if not values:
            continue
        num_layers += 1
        for idx, value in enumerate(values):
            out[idx % size] += float(value)
    if num_layers == 0:
        return [0.0] * size
    return [round(value / num_layers, 5) for value in out]


def _flatten_layer_projections(
    hermes_state: HermesState,
    *,
    projection_dim: int,
    max_layers: int,
) -> list[float]:
    selected = _select_layer_keys(hermes_state, max_layers=max_layers)
    vectors: list[float] = []
    for key in selected:
        vectors.extend(_fit_vector(hermes_state.layer_projections.get(key, []), size=projection_dim))
    missing_layers = max(0, max_layers - len(selected))
    if missing_layers:
        vectors.extend([0.0] * (missing_layers * projection_dim))
    return vectors


def _select_layer_keys(hermes_state: HermesState, *, max_layers: int) -> list[str]:
    if not hermes_state.layer_projections:
        return []
    preferred = hermes_state.metadata.get("layer_projection_keys")
    if isinstance(preferred, list) and preferred:
        ordered = [str(key) for key in preferred if key in hermes_state.layer_projections]
    else:
        ordered = sorted(hermes_state.layer_projections.keys())
    return ordered[:max_layers]


def _fit_vector(values: list[float], *, size: int) -> list[float]:
    fitted = [float(value) for value in values[:size]]
    return fitted + [0.0] * max(0, size - len(fitted))
