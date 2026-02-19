# DSPy-REPL

[![Tests](https://github.com/Archelunch/dspy-repl/actions/workflows/ci.yml/badge.svg)](https://github.com/Archelunch/dspy-repl/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-orange?logo=buy-me-a-coffee)](https://buymeacoffee.com/mike_pavlukhin)

> **Modular non-Python REPL engines for [DSPy](https://github.com/stanfordnlp/dspy) Recursive Language Models.**

`dspy-repl` is a modular package for non-Python REPL-based RLM engines compatible with [DSPy](https://github.com/stanfordnlp/dspy), inspired by the [Recursive Language Models paper](https://arxiv.org/abs/2512.24601).

## Scope

- Keeps Python `dspy.RLM` inside DSPy as the canonical Python implementation.
- Provides modular engines for:
  - `SchemeRLM`
  - `SQLRLM`
  - `HaskellRLM`
  - `JavaScriptRLM`
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
from dspy_repl import SchemeRLM, SQLRLM, HaskellRLM, JavaScriptRLM

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"))

scheme_rlm = SchemeRLM("context, query -> answer")
result = scheme_rlm(context="...", query="...")
print(result.answer)
print(result.trajectory)  # step-by-step REPL history

js_rlm = JavaScriptRLM("context, query -> answer")
js_result = js_rlm(context="...", query="...")
print(js_result.answer)
```

## SQLRLM

`SQLRLM` uses Python's built-in `sqlite3` as its REPL environment -- no external runtime needed. The LLM writes SQL to explore data, call tools, and produce results.

### Basic usage

```python
import dspy
from dspy_repl import SQLRLM

dspy.configure(lm=dspy.LM("openai/gpt-4o"))

rlm = SQLRLM("context, query -> answer")
result = rlm(context="...", query="...")
print(result.answer)
```

### Pre-loaded schemas

When working with relational data, you can pre-load a SQL schema so the LLM sees the table structure from iteration 1 instead of spending iterations writing `CREATE TABLE` statements:

```python
rlm = SQLRLM(
    "query -> answer",
    preload_sql="schema.sql",          # path to .sql file or raw SQL string
    db_path="data/my_project.db",      # persist to file (default: ":memory:")
    skip_variable_tables={"query"},    # don't create a table for this input
)
result = rlm(query="Find all active projects led by engineers")
```

**`preload_sql`** accepts either a file path (e.g. `"schema.sql"`) or a raw SQL string. The DDL is executed once at startup. All tables -- including pre-loaded ones -- are visible in the LLM's prompt with column types, row counts, foreign key relationships, and CHECK constraints.

**`db_path`** controls where the SQLite database lives. Use a file path for persistence across runs. When reopening an existing database, `preload_sql` detects that tables already exist and skips the DDL.

**`skip_variable_tables`** prevents specified input variables from being materialized as SQL tables. Useful for string or structured inputs that serve as context rather than queryable data. These appear as plain text in the prompt instead.

The LLM prompt shows the full schema from iteration 1:

```
Input variables:
- query: "Find all active projects led by engineers"

Database tables:
- departments (id TEXT, name TEXT, budget REAL) -- 5 rows
  CHECKs: budget >= 0
- employees (id TEXT, name TEXT, department_id TEXT, role TEXT) -- 12 rows
  FKs: department_id -> departments.id
  CHECKs: role IN ('engineer','manager','designer','analyst')
- projects (id TEXT, name TEXT, lead_id TEXT, status TEXT) -- 8 rows
  FKs: lead_id -> employees.id
  CHECKs: status IN ('active','completed','cancelled')
```

### Using the SQLInterpreter directly

The underlying `SQLInterpreter` supports the same features and can be used standalone:

```python
from dspy_repl.interpreters.sql_interpreter import SQLInterpreter

schema = """
CREATE TABLE authors (id TEXT PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE books (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    author_id TEXT NOT NULL REFERENCES authors(id),
    genre TEXT CHECK(genre IN ('fiction','nonfiction','poetry'))
);
"""

with SQLInterpreter(preload_sql=schema) as interp:
    # describe_tables() returns columns, row counts, FKs, and CHECKs
    for t in interp.describe_tables():
        print(t["name"], t.get("foreign_keys", []), t.get("checks", []))

    interp.execute("INSERT INTO authors VALUES ('a1', 'Tolkien');")
    interp.execute("INSERT INTO books VALUES ('b1', 'The Hobbit', 'a1', 'fiction');")
    print(interp.execute("SELECT * FROM books;"))
```

### Tools as SQL functions

Custom Python functions can be registered as SQLite UDFs, callable directly from SQL:

```python
def classify(text: str) -> str:
    return "positive" if "good" in text.lower() else "negative"

rlm = SQLRLM(
    "reviews -> summary",
    preload_sql="CREATE TABLE results (id INTEGER PRIMARY KEY, sentiment TEXT);",
    tools=[classify],
)
# The LLM can write: INSERT INTO results SELECT id, classify(text) FROM reviews;
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
3. Execute code in the target REPL (Scheme/Haskell/SQL/JavaScript).
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
- `JavaScriptRLM`: requires `node`

### Install REPL runtimes

If you want to run all REPL-based engines and benchmark comparisons (including Python `dspy.RLM`), install:

- Python REPL engine in benchmarks (`dspy.RLM`): `deno`
- Scheme REPL engine (`SchemeRLM`): `guile`
- Haskell REPL engine (`HaskellRLM`): `ghci` from GHC
- JavaScript REPL engine (`JavaScriptRLM`): `node`

macOS (Homebrew):

```bash
brew install deno guile ghc node
```

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y deno guile-3.0 ghc nodejs npm
```

Verify tools are available:

```bash
deno --version
guile --version
ghci --version
node --version
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

Run OOLONG-Pairs benchmarks:

```bash
python -m dspy_repl.benchmarks.oolong_pairs_runner --model "gemini/gemini-3-flash-preview" --languages "sql,scheme,js" --max-samples 20
```

Run S-NIAH synthetic scaling benchmarks:

```bash
python -m dspy_repl.benchmarks.niah_runner --languages "python,sql,scheme" --num-tasks 50 --context-lengths "8192,32768,131072"
```

Generate a single HTML analytics report (tables + Plotly charts + insights):

```bash
python -m dspy_repl.benchmarks.report_runner --run-dir benchmark_results/<run_id>
```

Compare several runs in one report:

```bash
python -m dspy_repl.benchmarks.report_runner --run-dirs benchmark_results/<id1>,benchmark_results/<id2>
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

Full benchmark usage guide: `BENCHMARKS.md`.

## Local validation before release

```bash
python -m build
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
python -m twine check --strict dist/*
```

## Backlog

- Add shared context with PostgreSQL/MySQL.
- Test shared context in a multi-agent environment.
- Extend benchmarks with additional long-context suites.
- Optimize REPL instructions with GEPA.

