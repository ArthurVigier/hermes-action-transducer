from __future__ import annotations

from pathlib import Path


ALL_BENCHMARK_MODES = ["vanilla", "compact", "rich", "per_layer", "full"]


def resolve_benchmark_modes(benchmark_mode: str, modes_csv: str = "") -> list[str]:
    if benchmark_mode == "complete":
        return list(ALL_BENCHMARK_MODES)

    if benchmark_mode == "pair":
        modes = _parse_modes_csv(modes_csv)
        if len(modes) != 2:
            raise ValueError("Pair benchmark requires exactly two modes in --modes.")
        return modes

    if benchmark_mode == "custom":
        modes = _parse_modes_csv(modes_csv)
        if not modes:
            raise ValueError("Custom benchmark requires at least one mode in --modes.")
        return modes

    raise ValueError(f"Unknown benchmark mode: {benchmark_mode}")


def make_feature_config(
    mode: str,
    *,
    rich_projection_dim: int,
    layer_summary_dim: int,
    per_layer_projection_dim: int,
    max_layer_projections: int,
) -> "FeatureConfig":
    from hermes_action_transducer.features import FeatureConfig

    return FeatureConfig(
        mode=mode,
        rich_projection_dim=rich_projection_dim,
        layer_summary_dim=layer_summary_dim,
        per_layer_projection_dim=per_layer_projection_dim,
        max_layer_projections=max_layer_projections,
    )


def run_feature_benchmark(
    *,
    dataset_path: str | Path,
    checkpoint_dir: str | Path,
    benchmark_mode: str,
    modes_csv: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_dim: int,
    device: str,
    rich_projection_dim: int,
    layer_summary_dim: int,
    per_layer_projection_dim: int,
    max_layer_projections: int,
) -> dict:
    from hermes_action_transducer.dataset import JSONLSupervisedDataset
    from hermes_action_transducer.training import TrainingConfig, evaluate_supervised, train_supervised

    dataset_path = Path(dataset_path)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    selected_modes = resolve_benchmark_modes(benchmark_mode, modes_csv)
    results = []
    for mode in selected_modes:
        feature_config = make_feature_config(
            mode,
            rich_projection_dim=rich_projection_dim,
            layer_summary_dim=layer_summary_dim,
            per_layer_projection_dim=per_layer_projection_dim,
            max_layer_projections=max_layer_projections,
        )
        dataset = JSONLSupervisedDataset(dataset_path, feature_config=feature_config)
        checkpoint_path = checkpoint_dir / f"{mode}.pt"
        train_metrics = train_supervised(
            dataset,
            str(checkpoint_path),
            config=TrainingConfig(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                hidden_dim=hidden_dim,
                device=device,
                feature_config=feature_config,
            ),
        )
        eval_metrics = evaluate_supervised(dataset, str(checkpoint_path), device=device)
        results.append(
            {
                "mode": mode,
                "feature_config": feature_config.to_dict(),
                "feature_dim": dataset.feature_dim,
                "checkpoint": str(checkpoint_path),
                "train_metrics": train_metrics,
                "eval_metrics": eval_metrics,
            }
        )

    baseline_mode = "vanilla" if any(result["mode"] == "vanilla" for result in results) else results[0]["mode"]
    baseline = next(result for result in results if result["mode"] == baseline_mode)
    comparisons = []
    for result in results:
        comparisons.append(
            {
                "mode": result["mode"],
                "vs": baseline_mode,
                "delta_mode_acc": round(
                    result["eval_metrics"]["mode_acc"] - baseline["eval_metrics"]["mode_acc"], 6
                ),
                "delta_tool_acc": round(
                    result["eval_metrics"]["tool_acc"] - baseline["eval_metrics"]["tool_acc"], 6
                ),
            }
        )

    return {
        "dataset": str(dataset_path),
        "benchmark_mode": benchmark_mode,
        "selected_modes": selected_modes,
        "baseline_mode": baseline_mode,
        "results": results,
        "comparisons": comparisons,
    }


def _parse_modes_csv(modes_csv: str) -> list[str]:
    modes = [mode.strip() for mode in modes_csv.split(",") if mode.strip()]
    unknown = [mode for mode in modes if mode not in ALL_BENCHMARK_MODES]
    if unknown:
        raise ValueError(f"Unknown modes: {', '.join(unknown)}")
    return modes
