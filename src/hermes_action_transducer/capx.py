from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any


CAPX_TIER_ORDER = ["S1", "S2", "S3", "S4", "M1", "M2", "M3", "M4"]

_SUMMARY_RE = re.compile(r"(?P<success>\d+\.\d+)/(?P<reward>-?\d+\.\d+)/(?P<completed>\d+)")
_TOTAL_TRIALS_RE = re.compile(r"Total number of trials:\s*(?P<total>\d+)")
_AVG_CODE_BLOCKS_RE = re.compile(r"Average code blocks:\s*(?P<value>-?\d+\.\d+)")
_AVG_REGENERATIONS_RE = re.compile(r"Average regenerations:\s*(?P<value>-?\d+\.\d+)")
_AVG_FINISHES_RE = re.compile(r"Average finishes:\s*(?P<value>-?\d+\.\d+)")
_ELAPSED_RE = re.compile(r"Elapsed time:\s*(?P<value>-?\d+\.\d+)\s*seconds")


@dataclass(frozen=True)
class CapXTierDefinition:
    tier: str
    interaction_mode: str
    grounding_mode: str
    abstraction_note: str
    config_suffix: str


CAPX_TIER_DEFINITIONS: dict[str, CapXTierDefinition] = {
    "S1": CapXTierDefinition(
        tier="S1",
        interaction_mode="single_turn",
        grounding_mode="privileged",
        abstraction_note="Highest-abstraction single-turn config inferred from official privileged filenames.",
        config_suffix="_privileged",
    ),
    "S2": CapXTierDefinition(
        tier="S2",
        interaction_mode="single_turn",
        grounding_mode="perception",
        abstraction_note="Standard single-turn config inferred from official baseline filenames.",
        config_suffix="",
    ),
    "S3": CapXTierDefinition(
        tier="S3",
        interaction_mode="single_turn",
        grounding_mode="perception",
        abstraction_note="Reduced-API single-turn config inferred from official reduced_api filenames.",
        config_suffix="_reduced_api",
    ),
    "S4": CapXTierDefinition(
        tier="S4",
        interaction_mode="single_turn",
        grounding_mode="perception",
        abstraction_note="Lowest-abstraction single-turn config inferred from official reduced_api_exampleless filenames.",
        config_suffix="_reduced_api_exampleless",
    ),
    "M1": CapXTierDefinition(
        tier="M1",
        interaction_mode="multi_turn",
        grounding_mode="perception",
        abstraction_note="Base multi-turn config inferred from official multiturn filenames.",
        config_suffix="_multiturn",
    ),
    "M2": CapXTierDefinition(
        tier="M2",
        interaction_mode="multi_turn",
        grounding_mode="visual_feedback",
        abstraction_note="Multi-turn visual-feedback config inferred from official multiturn_vf filenames.",
        config_suffix="_multiturn_vf",
    ),
    "M3": CapXTierDefinition(
        tier="M3",
        interaction_mode="multi_turn",
        grounding_mode="visual_differencing",
        abstraction_note="Multi-turn visual-differencing config inferred from official multiturn_vdm filenames.",
        config_suffix="_multiturn_vdm",
    ),
    "M4": CapXTierDefinition(
        tier="M4",
        interaction_mode="multi_turn",
        grounding_mode="visual_differencing",
        abstraction_note="Reduced-API multi-turn visual-differencing config inferred from official multiturn_vdm_reduced_api filenames.",
        config_suffix="_multiturn_vdm_reduced_api",
    ),
}


@dataclass(frozen=True)
class CapXConfigSpec:
    suite: str
    tier: str
    config_path: str
    interaction_mode: str
    grounding_mode: str
    abstraction_note: str
    tier_source: str = "heuristic_from_official_filename"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_capx_tiers(benchmark_mode: str, tiers_csv: str = "") -> list[str]:
    if benchmark_mode == "complete":
        return list(CAPX_TIER_ORDER)

    tiers = _parse_tiers_csv(tiers_csv)
    if benchmark_mode == "pair":
        if len(tiers) != 2:
            raise ValueError("CapX pair benchmark requires exactly two tiers in --tiers.")
        return tiers

    if benchmark_mode == "custom":
        if not tiers:
            raise ValueError("CapX custom benchmark requires at least one tier in --tiers.")
        return tiers

    raise ValueError(f"Unknown CapX benchmark mode: {benchmark_mode}")


def infer_capx_tier(config_path: str | Path) -> str | None:
    stem = Path(config_path).stem

    if "skill_lib" in stem or "/hillclimb/" in str(config_path):
        return None

    if stem.endswith("_multiturn_vdm_reduced_api"):
        return "M4"
    if stem.endswith("_multiturn_vdm"):
        return "M3"
    if stem.endswith("_multiturn_vf"):
        return "M2"
    if stem.endswith("_multiturn"):
        return "M1"
    if stem.endswith("_reduced_api_exampleless"):
        return "S4"
    if stem.endswith("_reduced_api"):
        return "S3"
    if stem.endswith("_privileged"):
        return "S1"
    if "multiturn" in stem:
        return None
    return "S2"


def discover_capx_suites(capx_root: str | Path) -> list[str]:
    env_configs_dir = Path(capx_root) / "env_configs"
    if not env_configs_dir.exists():
        raise FileNotFoundError(f"CapX env_configs directory not found: {env_configs_dir}")

    suites = []
    for child in sorted(env_configs_dir.iterdir()):
        if not child.is_dir():
            continue
        if child.name in {"human_oracle_code", "real"}:
            continue
        if any(infer_capx_tier(path) for path in child.glob("*.yaml")):
            suites.append(child.name)
    return suites


def discover_capx_suite_configs(capx_root: str | Path, suite: str) -> dict[str, CapXConfigSpec]:
    suite_dir = Path(capx_root) / "env_configs" / suite
    if not suite_dir.exists():
        raise FileNotFoundError(f"CapX suite directory not found: {suite_dir}")

    discovered: dict[str, CapXConfigSpec] = {}
    for path in sorted(suite_dir.glob("*.yaml")):
        tier = infer_capx_tier(path)
        if tier is None:
            continue
        tier_def = CAPX_TIER_DEFINITIONS[tier]
        discovered[tier] = CapXConfigSpec(
            suite=suite,
            tier=tier,
            config_path=str(path),
            interaction_mode=tier_def.interaction_mode,
            grounding_mode=tier_def.grounding_mode,
            abstraction_note=tier_def.abstraction_note,
        )
    return discovered


def resolve_capx_config_specs(
    *,
    capx_root: str | Path,
    suites_csv: str,
    benchmark_mode: str,
    tiers_csv: str,
    configs_csv: str = "",
) -> list[CapXConfigSpec]:
    if configs_csv.strip():
        specs: list[CapXConfigSpec] = []
        for raw_path in [item.strip() for item in configs_csv.split(",") if item.strip()]:
            path = Path(raw_path)
            if not path.is_absolute():
                path = Path(capx_root) / raw_path
            tier = infer_capx_tier(path) or "custom"
            tier_def = CAPX_TIER_DEFINITIONS.get(
                tier,
                CapXTierDefinition(
                    tier="custom",
                    interaction_mode="unknown",
                    grounding_mode="unknown",
                    abstraction_note="Custom config path supplied manually.",
                    config_suffix="",
                ),
            )
            specs.append(
                CapXConfigSpec(
                    suite=path.parent.name,
                    tier=tier_def.tier,
                    config_path=str(path),
                    interaction_mode=tier_def.interaction_mode,
                    grounding_mode=tier_def.grounding_mode,
                    abstraction_note=tier_def.abstraction_note,
                    tier_source="custom_path",
                )
            )
        return specs

    suites = [item.strip() for item in suites_csv.split(",") if item.strip()]
    if not suites:
        raise ValueError("At least one CapX suite must be provided in --suites.")

    selected_tiers = resolve_capx_tiers(benchmark_mode, tiers_csv)
    specs = []
    for suite in suites:
        available = discover_capx_suite_configs(capx_root, suite)
        missing = [tier for tier in selected_tiers if tier not in available]
        if missing:
            raise FileNotFoundError(f"CapX suite '{suite}' is missing tiers: {', '.join(missing)}")
        specs.extend(available[tier] for tier in selected_tiers)
    return specs


def build_capx_launch_command(
    *,
    capx_root: str | Path,
    spec: CapXConfigSpec,
    model: str,
    server_url: str,
    total_trials: int | None = None,
    num_workers: int | None = None,
    temperature: float | None = None,
    record_video: bool = False,
    use_oracle_code: bool = False,
) -> list[str]:
    capx_root = Path(capx_root)
    config_path = Path(spec.config_path)
    try:
        config_arg = str(config_path.relative_to(capx_root))
    except ValueError:
        config_arg = str(config_path)

    command = [
        "uv",
        "run",
        "--no-sync",
        "--active",
        "capx/envs/launch.py",
        "--config-path",
        config_arg,
        "--model",
        model,
        "--server-url",
        server_url,
    ]
    if total_trials is not None:
        command.extend(["--total-trials", str(total_trials)])
    if num_workers is not None:
        command.extend(["--num-workers", str(num_workers)])
    if temperature is not None:
        command.extend(["--temperature", str(temperature)])
    if record_video:
        command.extend(["--record-video", "True"])
    if use_oracle_code:
        command.extend(["--use-oracle-code", "True"])
    return command


def parse_capx_summary_text(text: str) -> dict[str, Any] | None:
    summary_match = None
    for match in _SUMMARY_RE.finditer(text):
        summary_match = match

    if summary_match is None:
        return None

    total_trials_match = _TOTAL_TRIALS_RE.search(text)
    avg_code_blocks_match = _AVG_CODE_BLOCKS_RE.search(text)
    avg_regenerations_match = _AVG_REGENERATIONS_RE.search(text)
    avg_finishes_match = _AVG_FINISHES_RE.search(text)
    elapsed_match = _ELAPSED_RE.search(text)

    total_trials = int(total_trials_match.group("total")) if total_trials_match else None
    completed = int(summary_match.group("completed"))
    task_completed_rate = (completed / total_trials) if total_trials else None

    return {
        "success_rate": float(summary_match.group("success")),
        "average_reward": float(summary_match.group("reward")),
        "task_completed_count": completed,
        "total_trials": total_trials,
        "task_completed_rate": task_completed_rate,
        "average_code_blocks": float(avg_code_blocks_match.group("value")) if avg_code_blocks_match else None,
        "average_regenerations": float(avg_regenerations_match.group("value")) if avg_regenerations_match else None,
        "average_finishes": float(avg_finishes_match.group("value")) if avg_finishes_match else None,
        "elapsed_seconds": float(elapsed_match.group("value")) if elapsed_match else None,
    }


def parse_capx_summary_file(path: str | Path) -> dict[str, Any] | None:
    return parse_capx_summary_text(Path(path).read_text(encoding="utf-8"))


def run_capx_benchmark(
    *,
    capx_root: str | Path,
    suites_csv: str,
    benchmark_mode: str,
    tiers_csv: str,
    model: str,
    server_url: str,
    results_out: str | Path,
    log_dir: str | Path | None = None,
    total_trials: int | None = None,
    num_workers: int | None = None,
    temperature: float | None = None,
    record_video: bool = False,
    use_oracle_code: bool = False,
    dry_run: bool = False,
    configs_csv: str = "",
) -> dict[str, Any]:
    capx_root = Path(capx_root).resolve()
    results_out = Path(results_out)
    log_dir = Path(log_dir) if log_dir is not None else results_out.parent / "capx_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    specs = resolve_capx_config_specs(
        capx_root=capx_root,
        suites_csv=suites_csv,
        benchmark_mode=benchmark_mode,
        tiers_csv=tiers_csv,
        configs_csv=configs_csv,
    )

    runs = []
    for spec in specs:
        command = build_capx_launch_command(
            capx_root=capx_root,
            spec=spec,
            model=model,
            server_url=server_url,
            total_trials=total_trials,
            num_workers=num_workers,
            temperature=temperature,
            record_video=record_video,
            use_oracle_code=use_oracle_code,
        )

        run_record: dict[str, Any] = {
            **spec.to_dict(),
            "command": command,
        }

        if dry_run:
            run_record["dry_run"] = True
            runs.append(run_record)
            continue

        completed = subprocess.run(
            command,
            cwd=capx_root,
            capture_output=True,
            text=True,
            env=os.environ.copy(),
        )
        combined_output = completed.stdout
        if completed.stderr:
            combined_output = f"{combined_output}\n{completed.stderr}".strip()

        log_name = _sanitize_for_filename(f"{spec.suite}_{spec.tier}") + ".log"
        log_path = log_dir / log_name
        log_path.write_text(combined_output, encoding="utf-8")

        run_record["returncode"] = completed.returncode
        run_record["log_path"] = str(log_path)
        run_record["metrics"] = parse_capx_summary_text(combined_output)
        runs.append(run_record)

    payload = {
        "capx_root": str(capx_root),
        "model": model,
        "server_url": server_url,
        "benchmark_mode": benchmark_mode,
        "suites": [item.strip() for item in suites_csv.split(",") if item.strip()],
        "selected_tiers": _infer_selected_tiers(runs),
        "dry_run": dry_run,
        "runs": runs,
        "aggregates": {
            "overall": _aggregate_capx_runs(runs),
            "by_tier": _aggregate_grouped_capx_runs(runs, "tier"),
            "by_suite": _aggregate_grouped_capx_runs(runs, "suite"),
        },
    }
    results_out.parent.mkdir(parents=True, exist_ok=True)
    results_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _aggregate_capx_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [run for run in runs if run.get("metrics")]
    if not valid:
        return {
            "count": 0,
            "success_rate_mean": None,
            "average_reward_mean": None,
            "task_completed_rate_mean": None,
            "task_completed_count_mean": None,
            "elapsed_seconds_mean": None,
        }

    def _metric(name: str) -> list[float]:
        values = []
        for run in valid:
            value = run["metrics"].get(name)
            if value is not None:
                values.append(float(value))
        return values

    return {
        "count": len(valid),
        "success_rate_mean": _safe_mean(_metric("success_rate")),
        "average_reward_mean": _safe_mean(_metric("average_reward")),
        "task_completed_rate_mean": _safe_mean(_metric("task_completed_rate")),
        "task_completed_count_mean": _safe_mean(_metric("task_completed_count")),
        "elapsed_seconds_mean": _safe_mean(_metric("elapsed_seconds")),
    }


def _aggregate_grouped_capx_runs(runs: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        group = str(run.get(key, "unknown"))
        groups.setdefault(group, []).append(run)
    return {group: _aggregate_capx_runs(items) for group, items in sorted(groups.items())}


def _parse_tiers_csv(tiers_csv: str) -> list[str]:
    tiers = [tier.strip().upper() for tier in tiers_csv.split(",") if tier.strip()]
    unknown = [tier for tier in tiers if tier not in CAPX_TIER_ORDER]
    if unknown:
        raise ValueError(f"Unknown CapX tiers: {', '.join(unknown)}")
    return tiers


def _safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _infer_selected_tiers(runs: list[dict[str, Any]]) -> list[str]:
    tiers = []
    for run in runs:
        tier = run.get("tier")
        if tier and tier not in tiers:
            tiers.append(tier)
    return tiers


def _sanitize_for_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")
