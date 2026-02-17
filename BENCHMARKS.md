# Benchmarks

## Oolong Runner

`dspy-repl` includes an Oolong-only benchmark runner with configurable languages,
structured logs, and trajectory artifact persistence.

Run it with:

```bash
python -m dspy_repl.benchmarks.oolong_runner --model "gemini/gemini-3-flash-preview" --languages "scheme,sql"
```

## Oolong-Pairs Runner

`dspy-repl` includes an OOLONG-Pairs runner that evaluates pairwise user-ID
aggregation tasks (F1 scoring over predicted pairs).

Run it with:

```bash
python -m dspy_repl.benchmarks.oolong_pairs_runner --model "gemini/gemini-3-flash-preview" --languages "sql,scheme,js" --max-samples 20
```

The runner uses the same artifact format as OOLONG and writes full
trajectories per engine/sample.

## S-NIAH Runner

`dspy-repl` includes a synthetic Single Needle-In-A-Haystack runner for
long-context retrieval scaling.

Run it with:

```bash
python -m dspy_repl.benchmarks.niah_runner --languages "python,sql,scheme" --num-tasks 50 --context-lengths "8192,32768,131072"
```

S-NIAH supports:

- `--num-tasks` tasks per context length
- `--context-lengths` comma-separated token lengths
- `--max-samples` is applied per context length (set `0` to disable cap)

In addition to default run artifacts, S-NIAH writes:

- `summary_by_context_length.json`
- `by_engine_context_length.csv`

## Configuration

You can provide configuration via CLI flags or an optional JSON config file.
CLI arguments take precedence over JSON values.

Example config file:

```json
{
  "model": {
    "model": "gemini/gemini-3-flash-preview",
    "temperature": 0.15,
    "max_tokens": 200000
  },
  "dataset": {
    "dataset_name": "oolongbench/oolong-real",
    "dataset_split": "train",
    "max_samples": 20,
    "seed": 42,
    "sample_id": null
  },
  "run": {
    "languages": ["python", "scheme", "haskell", "sql"],
    "max_iterations": 10,
    "max_llm_calls": 20,
    "engine_timeout_seconds": 240,
    "verbose": false
  },
  "niah_dataset": {
    "num_tasks": 50,
    "context_lengths": [8192, 16384, 32768, 65536, 131072, 262144]
  },
  "parallel": {
    "enabled": true,
    "backend": "multiprocessing",
    "max_workers": null
  },
  "artifacts": {
    "save_dir": "benchmark_results",
    "incremental_save": true
  }
}
```

Run with config:

```bash
python -m dspy_repl.benchmarks.oolong_runner --config ./benchmark.json
```

## Language Selection

Select languages with `--languages`:

- `python` (requires `deno`)
- `scheme` (requires `guile`)
- `haskell` (requires `ghci`)
- `sql` (sqlite in-process)

Example:

```bash
python -m dspy_repl.benchmarks.oolong_runner --languages "scheme,sql"
```

Missing language prerequisites are logged and skipped automatically.

## Parallel Execution

By default, selected languages are run in parallel per sample using
`multiprocessing` (`ProcessPoolExecutor`).

- Use `--no-parallel` to force sequential language execution.
- Use `--max-workers <N>` to cap worker processes per sample.
- With `max_workers` unset, the runner uses one worker per selected language.

Example:

```bash
python -m dspy_repl.benchmarks.oolong_runner --languages "scheme,sql,haskell" --max-workers 2
```

Notes:

- All artifact writes happen in the parent process for deterministic output files.
- Worker process startup has overhead; tiny runs may not speed up much.
- On larger samples and multiple language engines, parallel mode typically reduces wall-clock time.

## Observability and Artifacts

Each run creates a timestamped directory under `save_dir`, including:

- `benchmark.log` structured lifecycle logs
- `run_config.json` effective run configuration
- `incremental_results.jsonl` incremental per-sample records (when enabled)
- `results.jsonl` final per-sample records with trajectory diagnostics
- `summary.json` and `by_engine.csv` aggregate metrics
- `trajectory_stats.json` and `per_engine_trajectory_stats.json`
- `trajectories/<engine>/<sample_id>.json` full trajectories

This layout makes it easy to debug failures, inspect trajectories, and compare
language performance across runs.

## Analytics and Report Generation

Generate a polished single-file HTML report with tables, Plotly charts, and
auto-generated insights:

```bash
python -m dspy_repl.benchmarks.report_runner --run-dir benchmark_results/<run_id>
```

Compare multiple runs:

```bash
python -m dspy_repl.benchmarks.report_runner --run-dirs benchmark_results/<id1>,benchmark_results/<id2>
```

Use latest runs from `benchmark_results`:

```bash
python -m dspy_repl.benchmarks.report_runner --latest 5
```

By default, reports are written to:

- `benchmark_results/reports/report_<timestamp>.html`

You can override output:

```bash
python -m dspy_repl.benchmarks.report_runner --latest 3 --output benchmark_results/reports/my_report.html
```

The report includes:

- Engine comparison table (sortable in browser)
- Score / latency / success-rate charts
- Score-vs-latency frontier chart
- S-NIAH degradation curves (when `summary_by_context_length.json` is available)
- Deterministic insight bullets about winners, tradeoffs, and stability
