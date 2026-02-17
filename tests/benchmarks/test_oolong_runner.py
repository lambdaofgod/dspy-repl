from __future__ import annotations

import json
import logging
from concurrent.futures import Future

import pytest  # type: ignore[import-not-found]

from dspy_repl.benchmarks.config import (  # type: ignore[import-not-found]
    build_arg_parser,
    load_benchmark_config,
    parse_languages,
)
from dspy_repl.benchmarks.oolong_runner import (  # type: ignore[import-not-found]
    TaskResult,
    _run_task_across_languages,
    resolve_runnable_languages,
    summarize_results,
)


def test_parse_languages_accepts_csv_and_list() -> None:
    assert parse_languages("scheme,sql") == ("scheme", "sql")
    assert parse_languages(["python", "haskell"]) == ("python", "haskell")
    assert parse_languages("js") == ("js",)


def test_parse_languages_rejects_unknown_values() -> None:
    with pytest.raises(ValueError):
        parse_languages("ruby")


def test_load_benchmark_config_merges_cli_and_json(tmp_path) -> None:
    config_path = tmp_path / "benchmark.json"
    config_path.write_text(
        json.dumps(
            {
                "model": {"model": "file-model", "temperature": 0.5, "max_tokens": 1111},
                "dataset": {"dataset_name": "dataset/from/file", "max_samples": 5},
                "run": {"languages": ["scheme", "sql"], "max_iterations": 3, "max_llm_calls": 4},
                "parallel": {"enabled": False, "max_workers": 4},
                "artifacts": {"save_dir": "file_results", "incremental_save": False},
            }
        ),
        encoding="utf-8",
    )

    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--config",
            str(config_path),
            "--model",
            "cli-model",
            "--languages",
            "haskell",
            "--max-iterations",
            "9",
            "--parallel",
            "--max-workers",
            "2",
            "--incremental-save",
        ]
    )
    cfg = load_benchmark_config(args)

    assert cfg.model.model == "cli-model"
    assert cfg.model.temperature == 0.5
    assert cfg.dataset.dataset_name == "dataset/from/file"
    assert cfg.dataset.max_samples == 5
    assert cfg.run.languages == ("haskell",)
    assert cfg.run.max_iterations == 9
    assert cfg.run.max_llm_calls == 4
    assert cfg.parallel.enabled is True
    assert cfg.parallel.max_workers == 2
    assert cfg.artifacts.save_dir == "file_results"
    assert cfg.artifacts.incremental_save is True


def test_resolve_runnable_languages_filters_missing(monkeypatch) -> None:
    monkeypatch.setattr(
        "dspy_repl.benchmarks.oolong_runner._check_prerequisites",
        lambda: {"python": False, "scheme": True, "haskell": False, "sql": True, "js": True},
    )
    logger = logging.getLogger("test.resolve.languages")
    languages = resolve_runnable_languages(("python", "scheme", "sql"), logger)
    assert languages == ["scheme", "sql"]


def test_summarize_results_groups_by_engine() -> None:
    results = [
        TaskResult(
            sample_id="a",
            task_name="task-a",
            engine="scheme",
            answer="ok",
            iterations=2,
            elapsed_seconds=1.5,
            success=True,
            score=0.9,
        ),
        TaskResult(
            sample_id="b",
            task_name="task-b",
            engine="scheme",
            answer="",
            iterations=0,
            elapsed_seconds=0,
            success=False,
            error="boom",
        ),
    ]
    summary = summarize_results(results)
    assert summary["scheme"]["total"] == 2.0
    assert summary["scheme"]["successes"] == 1.0
    assert summary["scheme"]["success_rate"] == 0.5


class _ImmediateExecutor:
    def submit(self, fn, *args, **kwargs):  # type: ignore[no-untyped-def]
        fut: Future[TaskResult] = Future()
        fut.set_result(fn(*args, **kwargs))
        return fut


def test_run_task_across_languages_parallel_dispatch_preserves_language_order(monkeypatch) -> None:
    def _fake_worker(*, language, task, config):  # type: ignore[no-untyped-def]
        return TaskResult(
            sample_id=str(task["id"]),
            task_name=str(task["name"]),
            engine=language,
            answer=f"{language}-answer",
            iterations=1,
            elapsed_seconds=0.1,
            success=True,
            expected=str(task.get("expected")) if task.get("expected") is not None else None,
            score=1.0,
            trajectory=[],
        )

    monkeypatch.setattr("dspy_repl.benchmarks.oolong_runner._run_task_worker", _fake_worker)
    parser = build_arg_parser()
    args = parser.parse_args(["--parallel", "--languages", "sql,scheme"])
    config = load_benchmark_config(args)
    logger = logging.getLogger("test.parallel.dispatch")
    task = {"id": "sample-1", "name": "Task", "signature": "context, query -> answer", "inputs": {}, "expected": "x"}

    results = _run_task_across_languages(
        task=task,
        languages=["sql", "scheme"],
        config=config,
        logger=logger,
        executor=_ImmediateExecutor(),  # type: ignore[arg-type]
    )

    assert [r.engine for r in results] == ["sql", "scheme"]
    assert all(r.success for r in results)
