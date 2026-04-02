from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from hermes_action_transducer.constants import HORIZON_VOCAB, MODE_VOCAB, SPEED_VOCAB, TOOL_VOCAB
from hermes_action_transducer.encoder import SimpleHermesEncoder
from hermes_action_transducer.features import action_ir_target_summary, build_feature_vector
from hermes_action_transducer.models import ActionIR, HermesState, RobotObservation, RobotProfileSpec, TargetRef
from hermes_action_transducer.profiles import get_profile


@dataclass
class SupervisedExample:
    observation: RobotObservation
    profile: RobotProfileSpec
    hermes_state: HermesState
    action_ir: ActionIR


class JSONLSupervisedDataset(Dataset):
    def __init__(self, path: str | Path, *, encoder=None) -> None:
        self.path = Path(path)
        self.encoder = encoder or SimpleHermesEncoder()
        self.examples = [self._parse_line(line) for line in self.path.read_text().splitlines() if line.strip()]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        target = action_ir_target_summary(example.action_ir)
        feature_vector = build_feature_vector(example.hermes_state, example.observation, example.profile)
        return {
            "features": torch.tensor(feature_vector, dtype=torch.float32),
            "mode": torch.tensor(MODE_VOCAB.index(str(target["mode"])), dtype=torch.long),
            "tool": torch.tensor(TOOL_VOCAB.index(str(target["tool"])), dtype=torch.long),
            "horizon": torch.tensor(HORIZON_VOCAB.index(str(target["horizon"])), dtype=torch.long),
            "speed": torch.tensor(SPEED_VOCAB.index(str(target["speed"])), dtype=torch.long),
            "confidence": torch.tensor(float(target["confidence"]), dtype=torch.float32),
            "caution": torch.tensor(float(target["caution"]), dtype=torch.float32),
            "force_limit": torch.tensor(float(target["force_limit"]), dtype=torch.float32),
            "motion_latent": torch.tensor(target["motion_latent"], dtype=torch.float32),
            "safety_latent": torch.tensor(target["safety_latent"], dtype=torch.float32),
        }

    def _parse_line(self, raw_line: str) -> SupervisedExample:
        payload = json.loads(raw_line)
        profile = get_profile(str(payload["profile"]))
        observation = RobotObservation(**payload["observation"])

        hermes_payload = payload.get("hermes_state")
        if hermes_payload:
            hermes_state = HermesState(**hermes_payload)
        else:
            hermes_state = self.encoder.encode(observation, profile)

        action_payload = payload["action_ir"]
        targets = [TargetRef(**target) for target in action_payload.get("targets", [])]
        action_ir = ActionIR(
            mode=action_payload["mode"],
            subgoal=action_payload["subgoal"],
            targets=targets,
            constraints=action_payload.get("constraints", {}),
            affordances=action_payload.get("affordances", {}),
            skill_prior=action_payload.get("skill_prior", []),
            motion_latent=action_payload.get("motion_latent", []),
            safety_latent=action_payload.get("safety_latent", []),
            horizon=action_payload.get("horizon", "short"),
            confidence=float(action_payload.get("confidence", 0.0)),
            metadata=action_payload.get("metadata", {}),
        )
        return SupervisedExample(
            observation=observation,
            profile=profile,
            hermes_state=hermes_state,
            action_ir=action_ir,
        )
