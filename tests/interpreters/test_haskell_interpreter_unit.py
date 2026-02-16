from __future__ import annotations

import os

import pytest

from dspy_repl.core.code_interpreter import CodeInterpreterError
from dspy_repl.interpreters.haskell_interpreter import HaskellInterpreter


class _FakeProcess:
    def __init__(self, stdout) -> None:
        self.stdout = stdout

    def poll(self) -> None:
        return None


def test_readline_with_timeout_raises_for_stalled_output() -> None:
    read_fd, write_fd = os.pipe()
    reader = os.fdopen(read_fd, "r", encoding="utf-8", closefd=True)
    writer = os.fdopen(write_fd, "w", encoding="utf-8", closefd=True)
    try:
        interp = HaskellInterpreter(execute_timeout_seconds=0.01)
        interp._process = _FakeProcess(stdout=reader)  # type: ignore[assignment]
        with pytest.raises(CodeInterpreterError, match="Timed out"):
            interp._readline_with_timeout(0.01, "during unit test")
    finally:
        writer.close()
        reader.close()
