from __future__ import annotations

import json

import pytest  # type: ignore[import-not-found]

from dspy_repl.benchmarks.config import (  # type: ignore[import-not-found]
    build_arg_parser,
    load_benchmark_config,
    parse_languages,
)
from dspy_repl.benchmarks.oolong_runner import run_oolong_benchmark  # type: ignore[import-not-found]


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


def test_load_benchmark_config_allows_disabling_verbose_from_cli(tmp_path) -> None:
    config_path = tmp_path / "benchmark.json"
    config_path.write_text(
        json.dumps(
            {
                "run": {"verbose": True},
            }
        ),
        encoding="utf-8",
    )

    parser = build_arg_parser()
    args = parser.parse_args(["--config", str(config_path), "--no-verbose"])
    cfg = load_benchmark_config(args)

    assert cfg.run.verbose is False


def test_run_oolong_benchmark_delegates_to_base(monkeypatch, tmp_path) -> None:
    parser = build_arg_parser()
    args = parser.parse_args([])
    config = load_benchmark_config(args)

    called: dict[str, object] = {}

    def _fake_run_benchmark(*, config, dataset_name, task_loader, score_fn):  # type: ignore[no-untyped-def]
        called["config"] = config
        called["dataset_name"] = dataset_name
        called["task_loader"] = task_loader
        called["score_fn"] = score_fn
        return tmp_path

    monkeypatch.setattr("dspy_repl.benchmarks.oolong_runner.run_benchmark", _fake_run_benchmark)
    run_dir = run_oolong_benchmark(config)

    assert run_dir == tmp_path
    assert called["dataset_name"] == "oolong_trec"
