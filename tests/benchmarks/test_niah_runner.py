from __future__ import annotations

import json
import sys
import types

from dspy_repl.benchmarks.config import (  # type: ignore[import-not-found]
    build_arg_parser,
    load_benchmark_config,
    parse_context_lengths,
)
from dspy_repl.benchmarks.niah_runner import (  # type: ignore[import-not-found]
    _extract_context_length,
    _limit_tasks_per_context_length,
    _load_niah_tasks,
    _write_context_length_summary,
)


def test_parse_context_lengths_parses_csv() -> None:
    assert parse_context_lengths("8192,16384") == (8192, 16384)


def test_extract_context_length_from_sample_id() -> None:
    assert _extract_context_length("niah_32768_001") == 32768
    assert _extract_context_length("other_001") is None


def test_load_niah_tasks_uses_extended_config(monkeypatch) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(["--num-tasks", "2", "--context-lengths", "64,128"])
    config = load_benchmark_config(args)

    fake_module = types.SimpleNamespace(
        generate_niah_tasks=lambda **kwargs: [
            {"id": "niah_64_001", "inputs": {}, "signature": "context, query -> answer"},
            {"id": "niah_128_001", "inputs": {}, "signature": "context, query -> answer"},
            {"id": "niah_128_002", "inputs": {}, "signature": "context, query -> answer"},
        ]
    )
    monkeypatch.setattr("dspy_repl.benchmarks.niah_runner._ensure_sibling_dspy_package", lambda: None)
    monkeypatch.setitem(sys.modules, "benchmarks.datasets.niah", fake_module)

    tasks = _load_niah_tasks(config)
    assert len(tasks) == 3


def test_limit_tasks_per_context_length() -> None:
    tasks = [
        {"id": "niah_64_001"},
        {"id": "niah_64_002"},
        {"id": "niah_128_001"},
        {"id": "niah_128_002"},
    ]
    filtered = _limit_tasks_per_context_length(tasks, max_samples=1)
    assert [task["id"] for task in filtered] == ["niah_64_001", "niah_128_001"]


def test_write_context_length_summary_creates_artifacts(tmp_path) -> None:
    results_path = tmp_path / "results.jsonl"
    rows = [
        {
            "sample_id": "niah_8192_001",
            "engine": "sql",
            "success": True,
            "score": 1.0,
            "elapsed_seconds": 1.2,
            "iterations": 2,
        },
        {
            "sample_id": "niah_8192_002",
            "engine": "sql",
            "success": False,
            "score": 0.0,
            "elapsed_seconds": 0.0,
            "iterations": 0,
        },
    ]
    with results_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    _write_context_length_summary(tmp_path)

    assert (tmp_path / "summary_by_context_length.json").exists()
    assert (tmp_path / "by_engine_context_length.csv").exists()
