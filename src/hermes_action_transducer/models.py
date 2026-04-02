from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class RobotObservation:
    task: str
    state_text: str = ""
    image_path: Optional[str] = None
    proprio: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RobotProfileSpec:
    name: str
    description: str
    default_mode: str
    supported_modes: list[str]
    runtime_tools: list[str]
    preferred_constraints: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HermesState:
    summary_text: str
    thought_vector: list[float] = field(default_factory=list)
    intent_vector: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TargetRef:
    kind: str
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ActionIR:
    mode: str
    subgoal: str
    targets: list[TargetRef] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    affordances: dict[str, float] = field(default_factory=dict)
    skill_prior: list[str] = field(default_factory=list)
    motion_latent: list[float] = field(default_factory=list)
    safety_latent: list[float] = field(default_factory=list)
    horizon: str = "short"
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["targets"] = [target.to_dict() for target in self.targets]
        return data


@dataclass
class CompiledAction:
    command_type: str
    name: str
    arguments: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResult:
    profile: RobotProfileSpec
    hermes_state: HermesState
    action_ir: ActionIR
    compiled_action: CompiledAction

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "hermes_state": self.hermes_state.to_dict(),
            "action_ir": self.action_ir.to_dict(),
            "compiled_action": self.compiled_action.to_dict(),
        }


class HermesEncoder(ABC):
    @abstractmethod
    def encode(self, observation: RobotObservation, profile: RobotProfileSpec) -> HermesState:
        ...


class ActionTransducer(ABC):
    @abstractmethod
    def predict(
        self,
        hermes_state: HermesState,
        observation: RobotObservation,
        profile: RobotProfileSpec,
    ) -> ActionIR:
        ...


class EmbodimentCompiler(ABC):
    @abstractmethod
    def compile(
        self,
        action_ir: ActionIR,
        observation: RobotObservation,
        profile: RobotProfileSpec,
    ) -> CompiledAction:
        ...
