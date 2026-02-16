"""Benchmark helpers for reproducible experiment runs."""

from dspy_repl.benchmarks.oolong_runner import main as run_oolong_benchmark_cli
from dspy_repl.benchmarks.oolong_runner import run_oolong_benchmark

__all__ = ["run_oolong_benchmark", "run_oolong_benchmark_cli"]
