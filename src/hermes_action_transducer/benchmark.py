from __future__ import annotations

import time
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
    latency_samples: int = 64,
    latency_warmup: int = 5,
) -> dict:
    from hermes_action_transducer.dataset import JSONLSupervisedDataset
    from hermes_action_transducer.features import build_feature_vector
    from hermes_action_transducer.learned_transducer import ActionIRNet
    from hermes_action_transducer.training import TrainingConfig, evaluate_supervised, train_supervised
    import torch

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
        latency_metrics = benchmark_latency(
            dataset=dataset,
            checkpoint_path=str(checkpoint_path),
            device=device,
            samples=latency_samples,
            warmup=latency_warmup,
        )
        results.append(
            {
                "mode": mode,
                "feature_config": feature_config.to_dict(),
                "feature_dim": dataset.feature_dim,
                "checkpoint": str(checkpoint_path),
                "train_metrics": train_metrics,
                "eval_metrics": eval_metrics,
                "latency_metrics": latency_metrics,
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
                "delta_avg_end_to_end_latency_ms": _round_or_none(
                    _metric_delta(
                        result["latency_metrics"].get("avg_end_to_end_latency_ms"),
                        baseline["latency_metrics"].get("avg_end_to_end_latency_ms"),
                    )
                ),
                "delta_avg_model_forward_latency_ms": _round_or_none(
                    _metric_delta(
                        result["latency_metrics"].get("avg_model_forward_latency_ms"),
                        baseline["latency_metrics"].get("avg_model_forward_latency_ms"),
                    )
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


def benchmark_latency(
    *,
    dataset,
    checkpoint_path: str | Path,
    device: str,
    samples: int = 64,
    warmup: int = 5,
) -> dict:
    from hermes_action_transducer.features import build_feature_vector
    import torch

    payload = torch.load(checkpoint_path, map_location=device)
    model = ActionIRNet(
        input_dim=payload["config"]["input_dim"],
        hidden_dim=payload["config"]["hidden_dim"],
    ).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    total_examples = min(len(dataset.examples), max(1, samples))
    selected_examples = dataset.examples[:total_examples]

    feature_latencies = []
    forward_latencies = []
    end_to_end_latencies = []

    for idx, example in enumerate(selected_examples):
        _sync_device(device)
        start_e2e = time.perf_counter()

        start_feature = time.perf_counter()
        feature_vector = build_feature_vector(
            example.hermes_state,
            example.observation,
            example.profile,
            dataset.feature_config,
        )
        feature_tensor = torch.tensor(feature_vector, dtype=torch.float32, device=device).unsqueeze(0)
        _sync_device(device)
        feature_elapsed_ms = (time.perf_counter() - start_feature) * 1000.0

        start_forward = time.perf_counter()
        with torch.no_grad():
            model(feature_tensor)
        _sync_device(device)
        forward_elapsed_ms = (time.perf_counter() - start_forward) * 1000.0
        end_to_end_elapsed_ms = (time.perf_counter() - start_e2e) * 1000.0

        if idx >= warmup:
            feature_latencies.append(feature_elapsed_ms)
            forward_latencies.append(forward_elapsed_ms)
            end_to_end_latencies.append(end_to_end_elapsed_ms)

    if not end_to_end_latencies:
        return {
            "samples_measured": 0,
            "warmup_samples": min(warmup, total_examples),
            "avg_feature_build_latency_ms": None,
            "avg_model_forward_latency_ms": None,
            "avg_end_to_end_latency_ms": None,
            "p50_end_to_end_latency_ms": None,
            "p95_end_to_end_latency_ms": None,
        }

    return {
        "samples_measured": len(end_to_end_latencies),
        "warmup_samples": min(warmup, total_examples),
        "avg_feature_build_latency_ms": _mean(feature_latencies),
        "avg_model_forward_latency_ms": _mean(forward_latencies),
        "avg_end_to_end_latency_ms": _mean(end_to_end_latencies),
        "p50_end_to_end_latency_ms": _percentile(end_to_end_latencies, 50),
        "p95_end_to_end_latency_ms": _percentile(end_to_end_latencies, 95),
    }


def _parse_modes_csv(modes_csv: str) -> list[str]:
    modes = [mode.strip() for mode in modes_csv.split(",") if mode.strip()]
    unknown = [mode for mode in modes if mode not in ALL_BENCHMARK_MODES]
    if unknown:
        raise ValueError(f"Unknown modes: {', '.join(unknown)}")
    return modes


def _sync_device(device: str) -> None:
    import torch

    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6)


def _percentile(values: list[float], percentile: int) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return round(ordered[0], 6)
    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    value = ordered[lower] * (1.0 - weight) + ordered[upper] * weight
    return round(value, 6)


def _metric_delta(value: float | None, baseline: float | None) -> float | None:
    if value is None or baseline is None:
        return None
    return value - baseline


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 6)
