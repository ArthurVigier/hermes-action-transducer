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
FEATURE_DIM = 45
