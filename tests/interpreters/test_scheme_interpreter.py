from __future__ import annotations

import os
import shutil

import pytest

from dspy_repl.core.code_interpreter import CodeInterpreterError, FinalOutput
from dspy_repl.interpreters.scheme_interpreter import SchemeInterpreter

pytestmark = pytest.mark.skipif(
    shutil.which("guile") is None or os.getenv("RUN_REPL_RUNTIME_TESTS") != "1",
    reason="Guile runtime tests require guile and RUN_REPL_RUNTIME_TESTS=1",
)


def test_state_persists_between_execute_calls() -> None:
    with SchemeInterpreter() as interp:
        interp.execute("(define counter 0)")
        interp.execute("(set! counter (+ counter 5))")
        out = interp.execute("(display counter) (newline)")
        assert out.strip() == "5"


def test_variable_injection_types() -> None:
    with SchemeInterpreter() as interp:
        out = interp.execute(
            "(display (and (string? name) (number? count) (boolean? flag))) (newline)",
            variables={"name": "Alice", "count": 2, "flag": True},
        )
        assert out.strip() == "#t"


def test_typed_submit_and_arity_validation() -> None:
    with SchemeInterpreter(output_fields=[{"name": "answer"}, {"name": "confidence"}]) as interp:
        ok = interp.execute('(SUBMIT "done" 0.9)')
        assert isinstance(ok, FinalOutput)
        assert ok.output == {"answer": "done", "confidence": 0.9}

        with pytest.raises(CodeInterpreterError):
            interp.execute('(SUBMIT "missing_second_arg")')


def test_tool_call_bridge() -> None:
    def echo(text: str) -> str:
        return f"Echo: {text}"

    with SchemeInterpreter(tools={"echo": echo}) as interp:
        out = interp.execute('(display (echo "hello")) (newline)')
        assert "Echo: hello" in out


def test_error_classification() -> None:
    with SchemeInterpreter() as interp:
        with pytest.raises(SyntaxError):
            interp.execute("(define (bad")

        with pytest.raises(CodeInterpreterError):
            interp.execute("(display unknown_value)")
