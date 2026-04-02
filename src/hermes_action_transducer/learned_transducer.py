from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from hermes_action_transducer.constants import FEATURE_DIM, HORIZON_VOCAB, MODE_VOCAB, SPEED_VOCAB, TOOL_VOCAB
from hermes_action_transducer.features import build_feature_vector
from hermes_action_transducer.models import (
    ActionIR,
    ActionTransducer,
    HermesState,
    RobotObservation,
    RobotProfileSpec,
    TargetRef,
)
from hermes_action_transducer.transducer import _infer_subgoal, _infer_targets


INPUT_DIM = FEATURE_DIM


class ActionIRNet(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = 96) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mode_head = nn.Linear(hidden_dim, len(MODE_VOCAB))
        self.tool_head = nn.Linear(hidden_dim, len(TOOL_VOCAB))
        self.horizon_head = nn.Linear(hidden_dim, len(HORIZON_VOCAB))
        self.speed_head = nn.Linear(hidden_dim, len(SPEED_VOCAB))
        self.scalar_head = nn.Linear(hidden_dim, 3)
        self.motion_head = nn.Linear(hidden_dim, 6)
        self.safety_head = nn.Linear(hidden_dim, 6)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.backbone(features)
        scalars = self.scalar_head(hidden)
        return {
            "mode_logits": self.mode_head(hidden),
            "tool_logits": self.tool_head(hidden),
            "horizon_logits": self.horizon_head(hidden),
            "speed_logits": self.speed_head(hidden),
            "confidence": torch.sigmoid(scalars[..., 0]),
            "caution": torch.sigmoid(scalars[..., 1]),
            "force_limit": torch.sigmoid(scalars[..., 2]),
            "motion_latent": torch.tanh(self.motion_head(hidden)),
            "safety_latent": torch.tanh(self.safety_head(hidden)),
        }


@dataclass
class LearnedTransducerConfig:
    input_dim: int = INPUT_DIM
    hidden_dim: int = 96


class LearnedActionTransducer(ActionTransducer):
    def __init__(self, model: ActionIRNet) -> None:
        self.model = model.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path) -> "LearnedActionTransducer":
        payload = torch.load(checkpoint_path, map_location="cpu")
        config = LearnedTransducerConfig(**payload["config"])
        model = ActionIRNet(input_dim=config.input_dim, hidden_dim=config.hidden_dim)
        model.load_state_dict(payload["model_state"])
        model.eval()
        return cls(model)

    def predict(
        self,
        hermes_state: HermesState,
        observation: RobotObservation,
        profile: RobotProfileSpec,
    ) -> ActionIR:
        features = torch.tensor(
            build_feature_vector(hermes_state, observation, profile),
            dtype=torch.float32,
        ).unsqueeze(0)
        with torch.no_grad():
            outputs = self.model(features)
        mode = MODE_VOCAB[int(outputs["mode_logits"].argmax(dim=-1).item())]
        tool = TOOL_VOCAB[int(outputs["tool_logits"].argmax(dim=-1).item())]
        horizon = HORIZON_VOCAB[int(outputs["horizon_logits"].argmax(dim=-1).item())]
        speed = SPEED_VOCAB[int(outputs["speed_logits"].argmax(dim=-1).item())]
        confidence = float(outputs["confidence"].item())
        caution = float(outputs["caution"].item())
        force_limit = float(outputs["force_limit"].item())
        motion_latent = outputs["motion_latent"].squeeze(0).tolist()
        safety_latent = outputs["safety_latent"].squeeze(0).tolist()

        targets = _infer_targets(observation.task.lower())
        return ActionIR(
            mode=mode,
            subgoal=_infer_subgoal(observation.task, mode),
            targets=targets if targets else [TargetRef(kind="task_anchor", name="primary_target")],
            constraints={
                **profile.preferred_constraints,
                "speed": speed,
                "caution": round(caution, 4),
                "force_limit": round(force_limit, 4),
            },
            affordances={"reachable": 0.75, "safe": round(caution, 4)},
            skill_prior=_tool_prior(tool, profile),
            motion_latent=[round(x, 4) for x in motion_latent],
            safety_latent=[round(x, 4) for x in safety_latent],
            horizon=horizon,
            confidence=round(confidence, 4),
            metadata={
                "profile": profile.name,
                "source": "learned-transducer",
            },
        )


def save_checkpoint(path: str | Path, model: ActionIRNet, config: LearnedTransducerConfig) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": {
                "input_dim": config.input_dim,
                "hidden_dim": config.hidden_dim,
            },
            "model_state": model.state_dict(),
        },
        out,
    )


def _tool_prior(tool: str, profile: RobotProfileSpec) -> list[str]:
    ordered = [tool] + [name for name in profile.runtime_tools if name != tool]
    return ordered[:4]
