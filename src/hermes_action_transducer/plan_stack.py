from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from hermes_action_transducer.constants import HORIZON_VOCAB, MODE_VOCAB, TOOL_VOCAB
from hermes_action_transducer.dataset import JSONLSupervisedDataset
from hermes_action_transducer.features import FeatureConfig, action_ir_target_summary, build_feature_vector
from hermes_action_transducer.models import ActionIR


SPEED_TO_SCALAR = {
    "slow": -1.0,
    "normal": 0.0,
    "fast": 1.0,
}


def derive_plan_code(action_ir: ActionIR) -> str:
    summary = action_ir_target_summary(action_ir)
    return f"{summary['mode']}::{summary['tool']}::{summary['horizon']}"


def build_control_target(action_ir: ActionIR) -> list[float]:
    summary = action_ir_target_summary(action_ir)
    speed = SPEED_TO_SCALAR.get(str(summary["speed"]), 0.0)
    return [
        speed,
        float(summary["confidence"]),
        float(summary["caution"]),
        float(summary["force_limit"]),
        *[float(value) for value in summary["motion_latent"]],
        *[float(value) for value in summary["safety_latent"]],
    ]


def build_plan_discovery_vector(action_ir: ActionIR) -> list[float]:
    summary = action_ir_target_summary(action_ir)
    return [
        *build_control_target(action_ir),
        *[1.0 if summary["mode"] == mode else 0.0 for mode in MODE_VOCAB],
        *[1.0 if summary["tool"] == tool else 0.0 for tool in TOOL_VOCAB],
        *[1.0 if summary["horizon"] == horizon else 0.0 for horizon in HORIZON_VOCAB],
    ]


@dataclass(frozen=True)
class LearnedPlanCodebook:
    plan_vocab: list[str]
    assignment_indices: list[int]
    representative_codes: list[str]
    centroids: list[list[float]]
    cluster_sizes: list[int]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def discover_plan_codebook(
    action_irs: list[ActionIR],
    *,
    max_codes: int = 8,
    kmeans_iterations: int = 12,
) -> LearnedPlanCodebook:
    if not action_irs:
        raise ValueError("Plan code discovery requires at least one action IR.")

    vectors = [build_plan_discovery_vector(action_ir) for action_ir in action_irs]
    num_codes = min(max(1, max_codes), len(vectors), _count_unique_vectors(vectors))
    assignments, centroids = _run_kmeans(vectors, num_codes=num_codes, iterations=kmeans_iterations)

    representative_codes: list[str] = []
    cluster_sizes: list[int] = []
    for cluster_index in range(num_codes):
        members = [idx for idx, assignment in enumerate(assignments) if assignment == cluster_index]
        cluster_sizes.append(len(members))
        representative_codes.append(_representative_code(vectors, members, centroids[cluster_index], action_irs))

    return LearnedPlanCodebook(
        plan_vocab=[f"plan_{index:02d}" for index in range(num_codes)],
        assignment_indices=assignments,
        representative_codes=representative_codes,
        centroids=centroids,
        cluster_sizes=cluster_sizes,
    )


class PlanCodeDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        *,
        feature_config: FeatureConfig | None = None,
        codebook: LearnedPlanCodebook | None = None,
    ) -> None:
        self.feature_config = feature_config or FeatureConfig(mode="compact")
        self.base_dataset = JSONLSupervisedDataset(path, feature_config=self.feature_config)
        self.codebook = codebook or discover_plan_codebook([example.action_ir for example in self.base_dataset.examples])
        if len(self.codebook.assignment_indices) != len(self.base_dataset.examples):
            raise ValueError("Learned plan codebook must align with dataset length.")
        self.plan_vocab = self.codebook.plan_vocab
        self.plan_indices = self.codebook.assignment_indices
        self.feature_dim = self.base_dataset.feature_dim

    def __len__(self) -> int:
        return len(self.base_dataset.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.base_dataset.examples[index]
        features = build_feature_vector(
            example.hermes_state,
            example.observation,
            example.profile,
            self.feature_config,
        )
        return {
            "features": torch.tensor(features, dtype=torch.float32),
            "plan_code": torch.tensor(self.plan_indices[index], dtype=torch.long),
        }


class PlanConditionedControlDataset(Dataset):
    def __init__(self, path: str | Path, *, codebook: LearnedPlanCodebook) -> None:
        self.feature_config = FeatureConfig(mode="vanilla")
        self.base_dataset = JSONLSupervisedDataset(path, feature_config=self.feature_config)
        self.codebook = codebook
        if len(self.codebook.assignment_indices) != len(self.base_dataset.examples):
            raise ValueError("Learned plan codebook must align with dataset length.")
        self.plan_vocab = self.codebook.plan_vocab
        self.plan_indices = self.codebook.assignment_indices
        self.control_dim = len(build_control_target(self.base_dataset.examples[0].action_ir))
        self.feature_dim = self.base_dataset.feature_dim + len(self.plan_vocab)

    def __len__(self) -> int:
        return len(self.base_dataset.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.base_dataset.examples[index]
        base_features = build_feature_vector(
            example.hermes_state,
            example.observation,
            example.profile,
            self.feature_config,
        )
        plan_condition = [0.0] * len(self.plan_vocab)
        plan_condition[self.plan_indices[index]] = 1.0
        control_target = build_control_target(example.action_ir)
        return {
            "features": torch.tensor(base_features + plan_condition, dtype=torch.float32),
            "control": torch.tensor(control_target, dtype=torch.float32),
        }


class PlanRecognizerNet(nn.Module):
    def __init__(self, input_dim: int, plan_vocab_size: int, hidden_dim: int = 96) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.plan_head = nn.Linear(hidden_dim, plan_vocab_size)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.backbone(features)
        return {"plan_logits": self.plan_head(hidden)}


class PlanConditionedPolicyNet(nn.Module):
    def __init__(self, input_dim: int, control_dim: int, hidden_dim: int = 96) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.control_head = nn.Linear(hidden_dim, control_dim)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.backbone(features)
        return {"control": self.control_head(hidden)}


@dataclass
class PlanStackTrainingConfig:
    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 1e-3
    hidden_dim: int = 96
    device: str = "cpu"
    feature_config: FeatureConfig | None = None
    max_plan_codes: int = 8
    kmeans_iterations: int = 12


def train_plan_stack(
    dataset_path: str | Path,
    checkpoint_dir: str | Path,
    *,
    config: PlanStackTrainingConfig | None = None,
) -> dict[str, object]:
    config = config or PlanStackTrainingConfig()
    feature_config = config.feature_config or FeatureConfig(mode="compact")
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    base_dataset = JSONLSupervisedDataset(dataset_path, feature_config=feature_config)
    codebook = discover_plan_codebook(
        [example.action_ir for example in base_dataset.examples],
        max_codes=config.max_plan_codes,
        kmeans_iterations=config.kmeans_iterations,
    )

    plan_dataset = PlanCodeDataset(dataset_path, feature_config=feature_config, codebook=codebook)
    plan_checkpoint = checkpoint_dir / "plan_predictor.pt"
    plan_train_metrics = _train_plan_predictor(plan_dataset, plan_checkpoint, config=config)
    plan_eval_metrics = evaluate_plan_predictor(plan_dataset, plan_checkpoint, device=config.device)

    control_dataset = PlanConditionedControlDataset(dataset_path, codebook=codebook)
    control_checkpoint = checkpoint_dir / "control_policy.pt"
    control_train_metrics = _train_control_policy(control_dataset, control_checkpoint, config=config)
    control_eval_metrics = evaluate_control_policy(control_dataset, control_checkpoint, device=config.device)

    return {
        "dataset": str(dataset_path),
        "feature_config": feature_config.to_dict(),
        "plan_vocab": codebook.plan_vocab,
        "plan_codebook": codebook.to_dict(),
        "checkpoints": {
            "plan_predictor": str(plan_checkpoint),
            "control_policy": str(control_checkpoint),
        },
        "plan_metrics": {
            "train": plan_train_metrics,
            "eval": plan_eval_metrics,
        },
        "control_metrics": {
            "train": control_train_metrics,
            "eval": control_eval_metrics,
        },
    }


def evaluate_plan_predictor(dataset: PlanCodeDataset, checkpoint_path: str | Path, *, device: str = "cpu") -> dict[str, float]:
    payload = torch.load(checkpoint_path, map_location=device)
    model = PlanRecognizerNet(
        input_dim=payload["config"]["input_dim"],
        plan_vocab_size=len(payload["config"]["plan_vocab"]),
        hidden_dim=payload["config"]["hidden_dim"],
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    correct = 0.0
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["features"].to(device))
            labels = batch["plan_code"].to(device)
            correct += float((outputs["plan_logits"].argmax(dim=-1) == labels).float().sum().item())
            total += float(labels.shape[0])
    return {"plan_acc": correct / max(1.0, total)}


def evaluate_control_policy(
    dataset: PlanConditionedControlDataset,
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
) -> dict[str, float]:
    payload = torch.load(checkpoint_path, map_location=device)
    model = PlanConditionedPolicyNet(
        input_dim=payload["config"]["input_dim"],
        control_dim=payload["config"]["control_dim"],
        hidden_dim=payload["config"]["hidden_dim"],
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    total_mse = 0.0
    total = 0.0
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["features"].to(device))
            target = batch["control"].to(device)
            mse = torch.mean((outputs["control"] - target) ** 2, dim=-1)
            total_mse += float(mse.sum().item())
            total += float(target.shape[0])
    return {"control_mse": total_mse / max(1.0, total)}


def _train_plan_predictor(
    dataset: PlanCodeDataset,
    checkpoint_path: str | Path,
    *,
    config: PlanStackTrainingConfig,
) -> dict[str, float]:
    model = PlanRecognizerNet(
        input_dim=dataset.feature_dim,
        plan_vocab_size=len(dataset.plan_vocab),
        hidden_dim=config.hidden_dim,
    ).to(config.device)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    ce_loss = nn.CrossEntropyLoss()

    final_metrics = {"loss": 0.0, "plan_acc": 0.0}
    for _epoch in range(config.epochs):
        running = {"loss": 0.0, "correct": 0.0, "count": 0.0}
        model.train()
        for batch in loader:
            features = batch["features"].to(config.device)
            labels = batch["plan_code"].to(config.device)
            outputs = model(features)
            loss = ce_loss(outputs["plan_logits"], labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = float(labels.shape[0])
            running["loss"] += float(loss.item()) * batch_size
            running["correct"] += float((outputs["plan_logits"].argmax(dim=-1) == labels).float().sum().item())
            running["count"] += batch_size
        final_metrics = {
            "loss": running["loss"] / max(1.0, running["count"]),
            "plan_acc": running["correct"] / max(1.0, running["count"]),
        }

    _save_plan_predictor_checkpoint(
        checkpoint_path,
        model,
        input_dim=dataset.feature_dim,
        hidden_dim=config.hidden_dim,
        feature_config=dataset.feature_config,
        codebook=dataset.codebook,
    )
    return final_metrics


def _train_control_policy(
    dataset: PlanConditionedControlDataset,
    checkpoint_path: str | Path,
    *,
    config: PlanStackTrainingConfig,
) -> dict[str, float]:
    model = PlanConditionedPolicyNet(
        input_dim=dataset.feature_dim,
        control_dim=dataset.control_dim,
        hidden_dim=config.hidden_dim,
    ).to(config.device)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    mse_loss = nn.MSELoss()

    final_metrics = {"loss": 0.0}
    for _epoch in range(config.epochs):
        model.train()
        running = {"loss": 0.0, "count": 0.0}
        for batch in loader:
            features = batch["features"].to(config.device)
            target = batch["control"].to(config.device)
            outputs = model(features)
            loss = mse_loss(outputs["control"], target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = float(target.shape[0])
            running["loss"] += float(loss.item()) * batch_size
            running["count"] += batch_size
        final_metrics = {"loss": running["loss"] / max(1.0, running["count"])}

    _save_control_policy_checkpoint(
        checkpoint_path,
        model,
        input_dim=dataset.feature_dim,
        control_dim=dataset.control_dim,
        hidden_dim=config.hidden_dim,
        codebook=dataset.codebook,
    )
    return final_metrics


def _save_plan_predictor_checkpoint(
    path: str | Path,
    model: PlanRecognizerNet,
    *,
    input_dim: int,
    hidden_dim: int,
    feature_config: FeatureConfig,
    codebook: LearnedPlanCodebook,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": {
                "input_dim": input_dim,
                "hidden_dim": hidden_dim,
                "feature_config": feature_config.to_dict(),
                "plan_vocab": codebook.plan_vocab,
                "plan_codebook": codebook.to_dict(),
            },
            "model_state": model.state_dict(),
        },
        out,
    )


def _save_control_policy_checkpoint(
    path: str | Path,
    model: PlanConditionedPolicyNet,
    *,
    input_dim: int,
    control_dim: int,
    hidden_dim: int,
    codebook: LearnedPlanCodebook,
) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": {
                "input_dim": input_dim,
                "control_dim": control_dim,
                "hidden_dim": hidden_dim,
                "plan_vocab": codebook.plan_vocab,
                "plan_codebook": codebook.to_dict(),
            },
            "model_state": model.state_dict(),
        },
        out,
    )


def _count_unique_vectors(vectors: list[list[float]]) -> int:
    signatures = {tuple(round(value, 5) for value in vector) for vector in vectors}
    return max(1, len(signatures))


def _run_kmeans(vectors: list[list[float]], *, num_codes: int, iterations: int) -> tuple[list[int], list[list[float]]]:
    data = torch.tensor(vectors, dtype=torch.float32)
    centroids = _initial_centroids(data, num_codes=num_codes)
    assignments = torch.zeros(data.shape[0], dtype=torch.long)

    for _ in range(max(1, iterations)):
        distances = torch.cdist(data, centroids)
        assignments = distances.argmin(dim=1)
        updated = []
        for cluster_index in range(num_codes):
            mask = assignments == cluster_index
            if bool(mask.any()):
                updated.append(data[mask].mean(dim=0))
            else:
                updated.append(centroids[cluster_index])
        centroids = torch.stack(updated, dim=0)

    return assignments.tolist(), [[float(value) for value in row.tolist()] for row in centroids]


def _initial_centroids(data: torch.Tensor, *, num_codes: int) -> torch.Tensor:
    if num_codes == 1:
        return data[:1].clone()
    indices = torch.linspace(0, data.shape[0] - 1, steps=num_codes).round().long()
    return data[indices].clone()


def _representative_code(
    vectors: list[list[float]],
    member_indices: list[int],
    centroid: list[float],
    action_irs: list[ActionIR],
) -> str:
    if not member_indices:
        return "unassigned"
    centroid_tensor = torch.tensor(centroid, dtype=torch.float32)
    best_index = member_indices[0]
    best_distance = float("inf")
    for index in member_indices:
        vector_tensor = torch.tensor(vectors[index], dtype=torch.float32)
        distance = float(torch.sum((vector_tensor - centroid_tensor) ** 2).item())
        if distance < best_distance:
            best_distance = distance
            best_index = index
    return derive_plan_code(action_irs[best_index])
