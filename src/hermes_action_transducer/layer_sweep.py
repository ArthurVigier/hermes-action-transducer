from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from hermes_action_transducer.benchmark import run_feature_benchmark


def parse_layer_indices_csv(raw: str) -> list[int]:
    layers = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not layers:
        raise ValueError("Layer sweep requires at least one layer index.")
    deduped: list[int] = []
    for layer in layers:
        if layer not in deduped:
            deduped.append(layer)
    return deduped


def layer_tag(layer_index: int) -> str:
    return f"neg{abs(layer_index)}" if layer_index < 0 else f"pos{layer_index}"


def run_layer_sweep(
    *,
    root_dir: str | Path,
    dataset_id: str,
    split: str,
    robot_profile: str,
    source_format: str,
    model_id: str,
    device_map: str,
    torch_dtype: str,
    hf_cache_dir: str,
    local_files_only: bool,
    max_length: int,
    pool_strategy: str,
    rich_projection_dim: int,
    layer_projection_dim: int,
    additional_layer_indices: str,
    attn_implementation: str,
    max_rows: int,
    max_episodes: int,
    tasks_jsonl: str,
    streaming: bool,
    layer_indices: list[int],
    dataset_dir: str | Path,
    checkpoint_dir: str | Path,
    results_out: str | Path,
    force_rebuild_dataset: bool,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_dim: int,
    device: str,
    feature_rich_projection_dim: int,
    feature_layer_summary_dim: int,
    feature_per_layer_projection_dim: int,
    feature_max_layer_projections: int,
    latency_samples: int,
    latency_warmup: int,
    pair_modes_csv: str = "vanilla,compact",
) -> dict[str, Any]:
    root_dir = Path(root_dir)
    dataset_dir = Path(dataset_dir)
    checkpoint_dir = Path(checkpoint_dir)
    results_out = Path(results_out)
    dataset_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_out.parent.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    for layer_index in layer_indices:
        tag = layer_tag(layer_index)
        dataset_path = dataset_dir / f"{dataset_id.replace('/', '__')}_{tag}.jsonl"
        layer_checkpoint_dir = checkpoint_dir / tag
        layer_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        convert_cmd = [
            sys.executable,
            str(root_dir / "scripts" / "convert_hf_dataset.py"),
            "--dataset-id",
            dataset_id,
            "--split",
            split,
            "--robot-profile",
            robot_profile,
            "--source-format",
            source_format,
            "--encoder-backend",
            "hermes_hf",
            "--model-id",
            model_id,
            "--device-map",
            device_map,
            "--torch-dtype",
            torch_dtype,
            "--max-length",
            str(max_length),
            "--layer-index",
            str(layer_index),
            "--pool-strategy",
            pool_strategy,
            "--rich-projection-dim",
            str(rich_projection_dim),
            "--layer-projection-dim",
            str(layer_projection_dim),
            f"--additional-layer-indices={additional_layer_indices}",
            "--out",
            str(dataset_path),
            "--max-rows",
            str(max_rows),
            "--max-episodes",
            str(max_episodes),
        ]
        if hf_cache_dir:
            convert_cmd.extend(["--hf-cache-dir", hf_cache_dir])
        if local_files_only:
            convert_cmd.append("--local-files-only")
        if attn_implementation:
            convert_cmd.extend(["--attn-implementation", attn_implementation])
        if tasks_jsonl:
            convert_cmd.extend(["--tasks-jsonl", tasks_jsonl])
        if streaming:
            convert_cmd.append("--streaming")

        convert_result: dict[str, Any] = {
            "dataset_path": str(dataset_path),
            "reused_dataset": dataset_path.exists() and not force_rebuild_dataset,
            "layer_index": layer_index,
            "convert_command": convert_cmd,
        }
        if force_rebuild_dataset or not dataset_path.exists():
            completed = subprocess.run(convert_cmd, cwd=root_dir, capture_output=True, text=True)
            convert_result["convert_returncode"] = completed.returncode
            convert_result["convert_stdout"] = completed.stdout
            convert_result["convert_stderr"] = completed.stderr
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Layer sweep conversion failed for layer {layer_index}:\n{completed.stdout}\n{completed.stderr}"
                )

        benchmark_result = run_feature_benchmark(
            dataset_path=dataset_path,
            checkpoint_dir=layer_checkpoint_dir,
            benchmark_mode="pair",
            modes_csv=pair_modes_csv,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            device=device,
            rich_projection_dim=feature_rich_projection_dim,
            layer_summary_dim=feature_layer_summary_dim,
            per_layer_projection_dim=feature_per_layer_projection_dim,
            max_layer_projections=feature_max_layer_projections,
            latency_samples=latency_samples,
            latency_warmup=latency_warmup,
        )
        runs.append(
            {
                "layer_index": layer_index,
                "layer_tag": tag,
                **convert_result,
                "benchmark": benchmark_result,
            }
        )

    payload = {
        "dataset_id": dataset_id,
        "split": split,
        "robot_profile": robot_profile,
        "pair_modes": [item.strip() for item in pair_modes_csv.split(",") if item.strip()],
        "layer_indices": layer_indices,
        "runs": runs,
        "best_by_tool_acc": _best_run(runs, "tool"),
        "best_by_latency": _best_run(runs, "latency"),
    }
    results_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _best_run(runs: list[dict[str, Any]], metric: str) -> dict[str, Any] | None:
    candidates = []
    for run in runs:
        benchmark = run["benchmark"]
        compact_result = next((item for item in benchmark["results"] if item["mode"] == "compact"), None)
        if not compact_result:
            continue
        if metric == "tool":
            value = compact_result["eval_metrics"]["tool_acc"]
            score = (value, -compact_result["latency_metrics"]["avg_end_to_end_latency_ms"])
        else:
            value = compact_result["latency_metrics"]["avg_end_to_end_latency_ms"]
            score = (-value, compact_result["eval_metrics"]["tool_acc"])
        candidates.append((score, run["layer_index"], value, compact_result))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    _score, layer_index, value, compact_result = candidates[0]
    return {
        "layer_index": layer_index,
        "value": value,
        "compact_result": compact_result,
    }
