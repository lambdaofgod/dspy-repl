from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from dspy_repl.benchmarks.base_runner import _ensure_sibling_dspy_package, run_benchmark
from dspy_repl.benchmarks.config import BenchmarkConfig, build_arg_parser, load_benchmark_config

SAMPLE_ID_RE = re.compile(r"niah_(\d+)_\d+")


def _limit_tasks_per_context_length(tasks: list[dict[str, Any]], max_samples: int) -> list[dict[str, Any]]:
    if max_samples <= 0:
        return tasks

    counts: dict[int, int] = {}
    filtered: list[dict[str, Any]] = []
    for task in tasks:
        sample_id = str(task.get("id", ""))
        context_length = _extract_context_length(sample_id)
        if context_length is None:
            # Preserve non-standard IDs since we cannot bucket them safely.
            filtered.append(task)
            continue
        current = counts.get(context_length, 0)
        if current >= max_samples:
            continue
        counts[context_length] = current + 1
        filtered.append(task)
    return filtered


def _load_niah_tasks(config: BenchmarkConfig) -> list[dict[str, Any]]:
    _ensure_sibling_dspy_package()
    from benchmarks.datasets.niah import generate_niah_tasks  # type: ignore[import-not-found]

    tasks = generate_niah_tasks(
        num_tasks=config.niah_dataset.num_tasks,
        context_lengths=list(config.niah_dataset.context_lengths),
        seed=config.dataset.seed,
    )
    tasks = _limit_tasks_per_context_length(tasks, config.dataset.max_samples)
    if config.dataset.sample_id is None:
        return tasks
    filtered = [task for task in tasks if str(task.get("id")) == config.dataset.sample_id]
    if not filtered:
        raise ValueError(f"sample_id '{config.dataset.sample_id}' was not found in loaded S-NIAH samples.")
    return filtered


def _niah_score(predicted: str, expected: str) -> float:
    _ensure_sibling_dspy_package()
    from benchmarks.datasets.niah import niah_score_fn  # type: ignore[import-not-found]

    return float(niah_score_fn(predicted, expected))


def _extract_context_length(sample_id: str) -> int | None:
    match = SAMPLE_ID_RE.match(sample_id)
    if not match:
        return None
    return int(match.group(1))


def _write_context_length_summary(run_dir: Path) -> None:
    results_path = run_dir / "results.jsonl"
    if not results_path.exists():
        return

    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    with results_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            sample_id = str(row.get("sample_id", ""))
            length = _extract_context_length(sample_id)
            if length is None:
                continue
            key = (str(row.get("engine", "unknown")), length)
            grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for (engine, context_length), rows in sorted(grouped.items()):
        successes = [row for row in rows if bool(row.get("success", False))]
        scores = [float(row["score"]) for row in successes if row.get("score") is not None]
        latencies = [float(row.get("elapsed_seconds", 0.0)) for row in successes]
        iterations = [int(row.get("iterations", 0)) for row in successes]
        summary_rows.append(
            {
                "engine": engine,
                "context_length": context_length,
                "num_total": len(rows),
                "num_success": len(successes),
                "success_rate": len(successes) / len(rows) if rows else 0.0,
                "avg_score": (sum(scores) / len(scores)) if scores else 0.0,
                "avg_latency_seconds": (sum(latencies) / len(latencies)) if latencies else 0.0,
                "avg_iterations": (sum(iterations) / len(iterations)) if iterations else 0.0,
            }
        )

    payload = {"rows": summary_rows}
    (run_dir / "summary_by_context_length.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (run_dir / "by_engine_context_length.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "engine",
                "context_length",
                "num_total",
                "num_success",
                "success_rate",
                "avg_score",
                "avg_latency_seconds",
                "avg_iterations",
            ],
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def run_niah_benchmark(config: BenchmarkConfig) -> Path:
    run_dir = run_benchmark(
        config=config,
        dataset_name="s_niah",
        task_loader=_load_niah_tasks,
        score_fn=_niah_score,
    )
    _write_context_length_summary(run_dir)
    return run_dir


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    config = load_benchmark_config(args)
    run_dir = run_niah_benchmark(config)
    print(f"Artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
