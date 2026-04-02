from __future__ import annotations

import pytest

from hermes_action_transducer.benchmark import (
    _metric_delta,
    _percentile,
    make_feature_config,
    resolve_benchmark_modes,
)
from hermes_action_transducer.layer_sweep import layer_tag, parse_layer_indices_csv


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


def test_percentile_interpolates():
    assert _percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert _percentile([1.0, 2.0, 3.0, 4.0], 95) == 3.85


def test_metric_delta_handles_missing_values():
    assert _metric_delta(3.0, 1.5) == 1.5
    assert _metric_delta(None, 1.0) is None


def test_parse_layer_indices_csv_dedupes():
    assert parse_layer_indices_csv("-1,-2,-2,-4") == [-1, -2, -4]


def test_layer_tag():
    assert layer_tag(-8) == "neg8"
    assert layer_tag(3) == "pos3"
