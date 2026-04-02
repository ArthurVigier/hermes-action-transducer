from __future__ import annotations

from hermes_action_transducer.features import FeatureConfig, build_feature_vector, get_feature_dim
from hermes_action_transducer.models import HermesState, RobotObservation
from hermes_action_transducer.profiles import get_profile


def _sample_state() -> HermesState:
    return HermesState(
        summary_text="sample",
        thought_vector=[0.1] * 8,
        intent_vector=[0.2] * 8,
        hidden_projection=[0.3] * 40,
        layer_projections={
            "layer_-1": [0.4] * 20,
            "layer_-4": [0.5] * 20,
            "layer_-8": [0.6] * 20,
        },
        metadata={"layer_projection_keys": ["layer_-1", "layer_-4", "layer_-8"]},
    )


def test_feature_modes_have_expected_dims():
    observation = RobotObservation(task="Pick up the mug", state_text="mug left")
    profile = get_profile("arm")
    state = _sample_state()

    compact = FeatureConfig(mode="compact")
    rich = FeatureConfig(mode="rich", rich_projection_dim=32, layer_summary_dim=16)
    per_layer = FeatureConfig(mode="per_layer", rich_projection_dim=32, per_layer_projection_dim=12, max_layer_projections=3)

    assert len(build_feature_vector(state, observation, profile, compact)) == get_feature_dim(compact)
    assert len(build_feature_vector(state, observation, profile, rich)) == get_feature_dim(rich)
    assert len(build_feature_vector(state, observation, profile, per_layer)) == get_feature_dim(per_layer)
    assert get_feature_dim(compact) < get_feature_dim(rich) < get_feature_dim(per_layer)
