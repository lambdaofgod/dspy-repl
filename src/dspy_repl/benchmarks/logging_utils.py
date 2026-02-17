from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any


def _benchmark_formatter() -> logging.Formatter:
    return logging.Formatter(
        fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )


def setup_runtime_logger(*, verbose: bool, log_file: Path | None = None) -> logging.Logger:
    """
    Configure package-level logger used by language engines (`dspy_repl.*`).
    """
    logger = logging.getLogger("dspy_repl")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

    desired_log_file = str(log_file) if log_file is not None else None
    current_log_file = getattr(logger, "_dspy_repl_runtime_log_file", None)
    needs_rebuild = (not logger.handlers) or (current_log_file != desired_log_file)

    if needs_rebuild:
        logger.handlers.clear()
        formatter = _benchmark_formatter()

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file is not None:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        setattr(logger, "_dspy_repl_runtime_log_file", desired_log_file)
    else:
        for handler in logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(logging.DEBUG if verbose else logging.INFO)

    return logger


def setup_benchmark_logger(*, run_dir: Path, verbose: bool) -> logging.Logger:
    logger = logging.getLogger("dspy_repl.benchmarks")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

    formatter = _benchmark_formatter()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_file = run_dir / "benchmark.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    setup_runtime_logger(verbose=verbose, log_file=log_file)

    return logger


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    logger.info(json.dumps(payload, ensure_ascii=False, sort_keys=True))
