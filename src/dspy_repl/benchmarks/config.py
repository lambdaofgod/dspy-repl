from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, cast

Language = Literal["python", "scheme", "haskell", "sql"]
ALL_LANGUAGES: tuple[Language, ...] = ("python", "scheme", "haskell", "sql")
DEFAULT_MODEL = "gemini/gemini-3-flash-preview"


@dataclass(frozen=True)
class ModelConfig:
    model: str = DEFAULT_MODEL
    temperature: float = 0.15
    max_tokens: int = 100_000


@dataclass(frozen=True)
class DatasetConfig:
    dataset_name: str = "oolongbench/oolong-real"
    dataset_split: str = "train"
    max_samples: int = 20
    seed: int = 42
    sample_id: str | None = None


@dataclass(frozen=True)
class RunConfig:
    languages: tuple[Language, ...] = ALL_LANGUAGES
    max_iterations: int = 10
    max_llm_calls: int = 20
    engine_timeout_seconds: int = 240
    verbose: bool = False


@dataclass(frozen=True)
class ParallelConfig:
    enabled: bool = True
    backend: Literal["multiprocessing"] = "multiprocessing"
    max_workers: int | None = None


@dataclass(frozen=True)
class ArtifactConfig:
    save_dir: str = "benchmark_results"
    incremental_save: bool = True


@dataclass(frozen=True)
class BenchmarkConfig:
    model: ModelConfig = ModelConfig()
    dataset: DatasetConfig = DatasetConfig()
    run: RunConfig = RunConfig()
    parallel: ParallelConfig = ParallelConfig()
    artifacts: ArtifactConfig = ArtifactConfig()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["run"]["languages"] = list(self.run.languages)
        return payload


def parse_languages(raw: str | list[str] | tuple[str, ...] | None) -> tuple[Language, ...]:
    if raw is None:
        return ALL_LANGUAGES

    parts: list[str]
    if isinstance(raw, str):
        parts = raw.split(",")
    elif isinstance(raw, (list, tuple)):
        parts = []
        for item in raw:
            parts.extend(str(item).split(","))
    else:
        raise ValueError("Languages value must be a string or list of strings.")

    selected: list[Language] = []
    for token in parts:
        cleaned = token.strip().lower()
        if not cleaned:
            continue
        if cleaned not in ALL_LANGUAGES:
            choices = ", ".join(ALL_LANGUAGES)
            raise ValueError(f"Unsupported language '{cleaned}'. Supported values: {choices}")
        lang = cast(Language, cleaned)
        if lang not in selected:
            selected.append(lang)

    if not selected:
        raise ValueError("At least one language must be configured.")
    return tuple(selected)


def _read_json_config(path: str | None) -> dict[str, Any]:
    if path is None:
        return {}
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain an object at top level: {config_path}")
    return data


def _coalesce(cli_value: Any, file_value: Any, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if file_value is not None:
        return file_value
    return default


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Oolong benchmark across selected RLM languages.",
    )
    parser.add_argument("--config", help="Path to optional JSON config file")
    parser.add_argument("--model", default=None, help=f"LLM model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--temperature", type=float, default=None, help="LLM temperature")
    parser.add_argument("--max-tokens", type=int, default=None, help="LLM max tokens")
    parser.add_argument(
        "--languages",
        default=None,
        help="Comma-separated languages: python,scheme,haskell,sql",
    )
    parser.add_argument("--max-iterations", type=int, default=None, help="Max REPL iterations per run")
    parser.add_argument("--max-llm-calls", type=int, default=None, help="Max sub-LLM calls per run")
    parser.add_argument(
        "--engine-timeout-seconds",
        type=int,
        default=None,
        help="Per-engine timeout per sample; set 0 to disable",
    )
    parser.add_argument("--dataset-name", default=None, help="Hugging Face dataset name")
    parser.add_argument("--dataset-split", default=None, help="Dataset split")
    parser.add_argument("--max-samples", type=int, default=None, help="Max Oolong samples")
    parser.add_argument("--seed", type=int, default=None, help="Sampling seed")
    parser.add_argument("--sample-id", default=None, help="Only execute one sample id")
    parser.add_argument("--save-dir", default=None, help="Directory for benchmark artifacts")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logs")
    parser.add_argument(
        "--parallel",
        dest="parallel_enabled",
        action="store_true",
        help="Run selected languages in parallel per sample",
    )
    parser.add_argument(
        "--no-parallel",
        dest="parallel_enabled",
        action="store_false",
        help="Disable language-level parallel execution",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Worker process cap for parallel language execution per sample",
    )
    parser.add_argument(
        "--incremental-save",
        dest="incremental_save",
        action="store_true",
        help="Append per-sample JSONL records while benchmark is running",
    )
    parser.add_argument(
        "--no-incremental-save",
        dest="incremental_save",
        action="store_false",
        help="Only write aggregate artifacts when run completes",
    )
    parser.set_defaults(incremental_save=None, parallel_enabled=None)
    return parser


def load_benchmark_config(args: argparse.Namespace) -> BenchmarkConfig:
    file_config = _read_json_config(args.config)
    model_cfg = file_config.get("model", {})
    dataset_cfg = file_config.get("dataset", {})
    run_cfg = file_config.get("run", {})
    parallel_cfg = file_config.get("parallel", {})
    artifact_cfg = file_config.get("artifacts", {})

    model = ModelConfig(
        model=str(_coalesce(args.model, model_cfg.get("model"), DEFAULT_MODEL)),
        temperature=float(_coalesce(args.temperature, model_cfg.get("temperature"), 0.15)),
        max_tokens=int(_coalesce(args.max_tokens, model_cfg.get("max_tokens"), 100_000)),
    )
    dataset = DatasetConfig(
        dataset_name=str(_coalesce(args.dataset_name, dataset_cfg.get("dataset_name"), "oolongbench/oolong-real")),
        dataset_split=str(_coalesce(args.dataset_split, dataset_cfg.get("dataset_split"), "train")),
        max_samples=int(_coalesce(args.max_samples, dataset_cfg.get("max_samples"), 20)),
        seed=int(_coalesce(args.seed, dataset_cfg.get("seed"), 42)),
        sample_id=_coalesce(args.sample_id, dataset_cfg.get("sample_id"), None),
    )
    run = RunConfig(
        languages=parse_languages(_coalesce(args.languages, run_cfg.get("languages"), list(ALL_LANGUAGES))),
        max_iterations=int(_coalesce(args.max_iterations, run_cfg.get("max_iterations"), 10)),
        max_llm_calls=int(_coalesce(args.max_llm_calls, run_cfg.get("max_llm_calls"), 20)),
        engine_timeout_seconds=int(_coalesce(args.engine_timeout_seconds, run_cfg.get("engine_timeout_seconds"), 240)),
        verbose=bool(_coalesce(args.verbose if args.verbose else None, run_cfg.get("verbose"), False)),
    )
    parallel = ParallelConfig(
        enabled=bool(_coalesce(args.parallel_enabled, parallel_cfg.get("enabled"), True)),
        backend="multiprocessing",
        max_workers=(
            int(_coalesce(args.max_workers, parallel_cfg.get("max_workers"), None))
            if _coalesce(args.max_workers, parallel_cfg.get("max_workers"), None) is not None
            else None
        ),
    )
    artifacts = ArtifactConfig(
        save_dir=str(_coalesce(args.save_dir, artifact_cfg.get("save_dir"), "benchmark_results")),
        incremental_save=bool(_coalesce(args.incremental_save, artifact_cfg.get("incremental_save"), True)),
    )

    _validate_config(model=model, dataset=dataset, run=run, parallel=parallel)
    return BenchmarkConfig(model=model, dataset=dataset, run=run, parallel=parallel, artifacts=artifacts)


def _validate_config(*, model: ModelConfig, dataset: DatasetConfig, run: RunConfig, parallel: ParallelConfig) -> None:
    if not model.model.strip():
        raise ValueError("model.model must be a non-empty string")
    if model.max_tokens <= 0:
        raise ValueError("model.max_tokens must be > 0")
    if dataset.max_samples < 0:
        raise ValueError("dataset.max_samples must be >= 0")
    if run.max_iterations <= 0:
        raise ValueError("run.max_iterations must be > 0")
    if run.max_llm_calls <= 0:
        raise ValueError("run.max_llm_calls must be > 0")
    if run.engine_timeout_seconds < 0:
        raise ValueError("run.engine_timeout_seconds must be >= 0")
    if parallel.max_workers is not None and parallel.max_workers <= 0:
        raise ValueError("parallel.max_workers must be > 0 when set")
