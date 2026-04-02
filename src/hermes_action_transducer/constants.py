from __future__ import annotations


MODE_VOCAB = ["halt", "navigation", "tracking", "inspection", "manipulation"]
HORIZON_VOCAB = ["immediate", "short", "multi_step"]
SPEED_VOCAB = ["slow", "normal", "fast"]
TOOL_VOCAB = [
    "stop",
    "relative_move",
    "turn",
    "goto",
    "look_at",
    "follow_object",
    "pick",
    "place",
    "reach",
]
OBSERVATION_FEATURE_DIM = 29
BASE_FEATURE_DIM = 45
RICH_PROJECTION_DIM = 32
LAYER_SUMMARY_DIM = 16
PER_LAYER_PROJECTION_DIM = 64
MAX_LAYER_PROJECTIONS = 3
