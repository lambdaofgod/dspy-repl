# dspy-repl

`dspy-repl` is a modular package for non-Python REPL-based RLM engines compatible with DSPy.

## Scope

- Keeps Python `dspy.RLM` inside DSPy as the canonical Python implementation.
- Provides modular engines for:
  - `SchemeRLM`
  - `SQLRLM`
  - `HaskellRLM`
- Exposes extension points for adding new REPL languages.

## Install

```bash
pip install dspy-repl
```

For local development:

```bash
pip install -e ".[dev]"
```

## Quick usage

```python
import dspy
from dspy_repl import SchemeRLM, SQLRLM, HaskellRLM

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

scheme_rlm = SchemeRLM("context, query -> answer")
result = scheme_rlm(context="...", query="...")
print(result.answer)
print(result.trajectory)  # step-by-step REPL history
```

## Observability and debugging

`dspy-repl` is designed to expose what happened inside an RLM run:

- `result.trajectory` contains the full iterative REPL trace.
- Each trajectory step includes:
  - `reasoning`: model reasoning for that step
  - `code`: code sent to the language REPL
  - `output`: interpreter output/error text
- `SQLRLM` additionally exposes `last_sql_profile` timing breakdowns after each run.

Enable verbose engine logs:

```python
scheme_rlm = SchemeRLM("context, query -> answer", verbose=True)
```

With `verbose=True`, each iteration is logged with reasoning/code/output previews, which is useful for prompt/tool/debug loops.

## What happens inside an RLM

At a high level, each RLM run follows this loop:

1. Build REPL variable metadata from inputs.
2. Generate next action (reasoning + code) from the LM.
3. Execute code in the target REPL (Scheme/Haskell/SQL).
4. Append `{reasoning, code, output}` to trajectory.
5. Repeat until final output is submitted or max iterations is reached.
6. If max iterations is reached, run fallback extraction from accumulated trajectory.

This loop is shared in `dspy_repl.core.base_rlm` and specialized by language-specific wrappers.

## Architecture

- `dspy_repl.core`: shared execution loop and shared tool plumbing
- `dspy_repl.languages`: language-specific prompt templates and wrappers
- `dspy_repl.interpreters`: interpreter adapter exports
- `dspy_repl.compat`: thin compatibility shims for DSPy touchpoints

## DSPy compatibility

`dspy>=3.0.0`.

## Runtime prerequisites

- `SQLRLM`: no external runtime (uses Python `sqlite3`)
- `SchemeRLM`: requires `guile`
- `HaskellRLM`: requires `ghci` (GHC)

### Install REPL runtimes

If you want to run all REPL-based engines and benchmark comparisons (including Python `dspy.RLM`), install:

- Python REPL engine in benchmarks (`dspy.RLM`): `deno`
- Scheme REPL engine (`SchemeRLM`): `guile`
- Haskell REPL engine (`HaskellRLM`): `ghci` from GHC

macOS (Homebrew):

```bash
brew install deno guile ghc
```

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y deno guile-3.0 ghc
```

Verify tools are available:

```bash
deno --version
guile --version
ghci --version
```

### Python package dependencies for benchmarks

For Oolong benchmarks, you also need:

- `dspy-repl` (this package)
- `dspy`
- `datasets` (Hugging Face datasets loader used by Oolong adapter)

Example:

```bash
pip install -e ".[dev]" datasets
```

## Benchmarking (Oolong dataset)

The repository includes an Oolong benchmark runner with artifact saving and trajectory diagnostics.

Run benchmarks:

```bash
python -m dspy_repl.benchmarks.oolong_runner --model "gemini/gemini-3-flash-preview" --languages "python,scheme,sql,haskell"
```

### Multiprocessing

By default, selected languages run in parallel per sample using `multiprocessing`.

- Enable explicitly: `--parallel`
- Disable: `--no-parallel`
- Cap processes: `--max-workers 2`

Example:

```bash
python -m dspy_repl.benchmarks.oolong_runner --languages "scheme,sql,haskell" --max-workers 2
```

### Useful benchmark flags

- `--max-samples 20`
- `--sample-id <id>`
- `--engine-timeout-seconds 240`
- `--verbose`
- `--save-dir benchmark_results`
- `--config ./benchmark.json`

### Where results are saved

Each run creates a timestamped directory under `save_dir` with:

- `benchmark.log`: structured lifecycle logs
- `run_config.json`: effective run config
- `incremental_results.jsonl`: live per-sample writes (if enabled)
- `results.jsonl`: per-sample records with trajectory diagnostics
- `summary.json` and `by_engine.csv`: aggregate metrics
- `trajectory_stats.json` and `per_engine_trajectory_stats.json`
- `trajectories/<engine>/<sample_id>.json`: full trajectories

To inspect one execution deeply, start with a trajectory file and then correlate with the same sample in `results.jsonl` and `benchmark.log`.

Full benchmark usage guide: `docs/BENCHMARKS.md`.

## Local validation before release

```bash
python -m build
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
python -m twine check --strict dist/*
```

Release guide: `docs/RELEASING.md`.

