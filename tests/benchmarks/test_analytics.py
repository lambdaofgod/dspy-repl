from __future__ import annotations

import json

from dspy_repl.benchmarks.analytics import (  # type: ignore[import-not-found]
    analyze_runs,
    derive_metrics,
    generate_insights,
    load_run_artifacts,
    normalize_run_rows,
)


def _write_run_artifacts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    summary = {
        "run_id": "20260217T000000Z",
        "dataset": "oolong_trec",
        "rows": [
            {
                "engine": "sql",
                "success_rate": 0.8,
                "num_success": 8,
                "num_total": 10,
                "avg_latency_seconds": 12.5,
                "avg_iterations": 4.0,
                "avg_score": 0.72,
            }
        ],
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (tmp_path / "results.jsonl").write_text("", encoding="utf-8")
    per_engine = {
        "sql": {
            "failed_samples": 2,
            "timeout_samples": 1,
            "avg_error_steps": 0.2,
        }
    }
    (tmp_path / "per_engine_trajectory_stats.json").write_text(json.dumps(per_engine), encoding="utf-8")


def test_load_and_normalize_run_rows(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _write_run_artifacts(tmp_path)
    payload = load_run_artifacts(tmp_path)
    rows = normalize_run_rows(payload)
    assert len(rows) == 1
    row = rows[0]
    assert row["engine"] == "sql"
    assert row["dataset"] == "oolong_trec"
    assert row["num_total"] == 10


def test_derive_metrics_and_insights(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _write_run_artifacts(tmp_path)
    payload = load_run_artifacts(tmp_path)
    rows = normalize_run_rows(payload)
    derived = derive_metrics(rows)
    assert derived[0]["score_per_second"] > 0
    insights = generate_insights(derived)
    assert any("Best quality" in item for item in insights)


def test_analyze_runs_contract(tmp_path) -> None:  # type: ignore[no-untyped-def]
    _write_run_artifacts(tmp_path)
    analysis = analyze_runs([tmp_path])
    assert "derived_rows" in analysis
    assert "insights" in analysis
    assert len(analysis["derived_rows"]) == 1
