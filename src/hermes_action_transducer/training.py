from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader

from hermes_action_transducer.features import FeatureConfig
from hermes_action_transducer.learned_transducer import ActionIRNet, LearnedTransducerConfig, save_checkpoint


@dataclass
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 1e-3
    hidden_dim: int = 96
    device: str = "cpu"
    feature_config: FeatureConfig | None = None


def train_supervised(dataset, checkpoint_path: str, *, config: TrainingConfig | None = None) -> dict[str, float]:
    config = config or TrainingConfig()
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    feature_config = config.feature_config or getattr(dataset, "feature_config", FeatureConfig())
    model = ActionIRNet(input_dim=dataset.feature_dim, hidden_dim=config.hidden_dim).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    final_metrics: dict[str, float] = {}
    for _epoch in range(config.epochs):
        model.train()
        running = {
            "loss": 0.0,
            "mode_acc": 0.0,
            "tool_acc": 0.0,
            "count": 0.0,
        }
        for batch in loader:
            features = batch["features"].to(config.device)
            outputs = model(features)
            loss = (
                ce_loss(outputs["mode_logits"], batch["mode"].to(config.device))
                + ce_loss(outputs["tool_logits"], batch["tool"].to(config.device))
                + ce_loss(outputs["horizon_logits"], batch["horizon"].to(config.device))
                + ce_loss(outputs["speed_logits"], batch["speed"].to(config.device))
                + mse_loss(outputs["confidence"], batch["confidence"].to(config.device))
                + mse_loss(outputs["caution"], batch["caution"].to(config.device))
                + mse_loss(outputs["force_limit"], batch["force_limit"].to(config.device))
                + mse_loss(outputs["motion_latent"], batch["motion_latent"].to(config.device))
                + mse_loss(outputs["safety_latent"], batch["safety_latent"].to(config.device))
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = float(features.shape[0])
            running["loss"] += float(loss.item()) * batch_size
            running["mode_acc"] += float(
                (outputs["mode_logits"].argmax(dim=-1) == batch["mode"].to(config.device)).float().sum().item()
            )
            running["tool_acc"] += float(
                (outputs["tool_logits"].argmax(dim=-1) == batch["tool"].to(config.device)).float().sum().item()
            )
            running["count"] += batch_size

        final_metrics = {
            "loss": running["loss"] / max(1.0, running["count"]),
            "mode_acc": running["mode_acc"] / max(1.0, running["count"]),
            "tool_acc": running["tool_acc"] / max(1.0, running["count"]),
        }

    save_checkpoint(
        checkpoint_path,
        model,
        LearnedTransducerConfig(
            input_dim=dataset.feature_dim,
            hidden_dim=config.hidden_dim,
            feature_config=feature_config.to_dict(),
        ),
    )
    return final_metrics


def evaluate_supervised(dataset, checkpoint_path: str, *, device: str = "cpu") -> dict[str, float]:
    from hermes_action_transducer.learned_transducer import ActionIRNet

    payload = torch.load(checkpoint_path, map_location=device)
    model = ActionIRNet(
        input_dim=payload["config"]["input_dim"],
        hidden_dim=payload["config"]["hidden_dim"],
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    loader = DataLoader(dataset, batch_size=16, shuffle=False)

    totals = {"mode_acc": 0.0, "tool_acc": 0.0, "count": 0.0}
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["features"].to(device))
            batch_size = float(batch["features"].shape[0])
            totals["mode_acc"] += float(
                (outputs["mode_logits"].argmax(dim=-1) == batch["mode"].to(device)).float().sum().item()
            )
            totals["tool_acc"] += float(
                (outputs["tool_logits"].argmax(dim=-1) == batch["tool"].to(device)).float().sum().item()
            )
            totals["count"] += batch_size

    return {
        "mode_acc": totals["mode_acc"] / max(1.0, totals["count"]),
        "tool_acc": totals["tool_acc"] / max(1.0, totals["count"]),
    }
