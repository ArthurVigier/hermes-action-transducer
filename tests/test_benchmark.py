from __future__ import annotations

import pytest

from hermes_action_transducer.benchmark import make_feature_config, resolve_benchmark_modes


def test_resolve_complete_modes():
    assert resolve_benchmark_modes("complete") == ["vanilla", "compact", "rich", "per_layer", "full"]


def test_resolve_pair_modes():
    assert resolve_benchmark_modes("pair", "vanilla,full") == ["vanilla", "full"]


def test_resolve_pair_modes_requires_two():
    with pytest.raises(ValueError):
        resolve_benchmark_modes("pair", "vanilla")


def test_make_feature_config():
    config = make_feature_config(
        "per_layer",
        rich_projection_dim=32,
        layer_summary_dim=16,
        per_layer_projection_dim=64,
        max_layer_projections=3,
    )
    assert config.mode == "per_layer"
    assert config.max_layer_projections == 3
