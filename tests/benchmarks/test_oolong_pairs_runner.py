from __future__ import annotations

import sys
import types

from dspy_repl.benchmarks.config import build_arg_parser, load_benchmark_config  # type: ignore[import-not-found]
from dspy_repl.benchmarks.oolong_pairs_runner import (  # type: ignore[import-not-found]
    _load_oolong_pairs_tasks,
    run_oolong_pairs_benchmark,
)


def test_load_oolong_pairs_filters_sample(monkeypatch) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["--sample-id", "oolong_pairs_02"])
    config = load_benchmark_config(args)

    fake_module = types.SimpleNamespace(
        load_oolong_pairs_tasks=lambda **kwargs: [
            {"id": "oolong_pairs_01", "inputs": {}, "signature": "context, query -> answer"},
            {"id": "oolong_pairs_02", "inputs": {}, "signature": "context, query -> answer"},
        ]
    )
    monkeypatch.setattr("dspy_repl.benchmarks.oolong_pairs_runner._ensure_sibling_dspy_package", lambda: None)
    monkeypatch.setitem(sys.modules, "benchmarks.datasets.oolong_pairs", fake_module)

    tasks = _load_oolong_pairs_tasks(config)
    assert len(tasks) == 1
    assert tasks[0]["id"] == "oolong_pairs_02"


def test_run_oolong_pairs_benchmark_delegates_to_base(monkeypatch, tmp_path) -> None:
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

    monkeypatch.setattr("dspy_repl.benchmarks.oolong_pairs_runner.run_benchmark", _fake_run_benchmark)
    run_dir = run_oolong_pairs_benchmark(config)

    assert run_dir == tmp_path
    assert called["dataset_name"] == "oolong_pairs"
