from __future__ import annotations

from pathlib import Path
from typing import Any

from dspy_repl.benchmarks.base_runner import _ensure_sibling_dspy_package, run_benchmark
from dspy_repl.benchmarks.config import BenchmarkConfig, build_arg_parser, load_benchmark_config


def _load_oolong_pairs_tasks(config: BenchmarkConfig) -> list[dict[str, Any]]:
    _ensure_sibling_dspy_package()
    from benchmarks.datasets.oolong_pairs import load_oolong_pairs_tasks  # type: ignore[import-not-found]

    tasks = load_oolong_pairs_tasks(
        max_samples=config.dataset.max_samples,
        seed=config.dataset.seed,
        dataset_name=config.dataset.dataset_name,
        split=config.dataset.dataset_split,
    )
    if config.dataset.sample_id is None:
        return tasks
    filtered = [task for task in tasks if str(task.get("id")) == config.dataset.sample_id]
    if not filtered:
        raise ValueError(f"sample_id '{config.dataset.sample_id}' was not found in loaded OOLONG-Pairs samples.")
    return filtered


def _oolong_pairs_score(predicted: str, expected: str) -> float:
    _ensure_sibling_dspy_package()
    from benchmarks.datasets.oolong_pairs import oolong_pairs_score_fn  # type: ignore[import-not-found]

    return float(oolong_pairs_score_fn(predicted, expected))


def run_oolong_pairs_benchmark(config: BenchmarkConfig) -> Path:
    return run_benchmark(
        config=config,
        dataset_name="oolong_pairs",
        task_loader=_load_oolong_pairs_tasks,
        score_fn=_oolong_pairs_score,
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_benchmark_config(args)
    run_dir = run_oolong_pairs_benchmark(config)
    print(f"Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
