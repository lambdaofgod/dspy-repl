from __future__ import annotations

import os
import shutil
import time

import pytest

from dspy_repl.core.code_interpreter import FinalOutput
from dspy_repl.interpreters.haskell_interpreter import HaskellInterpreter

pytestmark = pytest.mark.skipif(
    shutil.which("ghci") is None or os.getenv("RUN_REPL_RUNTIME_TESTS") != "1",
    reason="GHCi runtime tests require ghci and RUN_REPL_RUNTIME_TESTS=1",
)


def test_prelude_loads_and_basic_execution() -> None:
    with HaskellInterpreter() as interp:
        out = interp.execute("print (1 + 1 :: Int)")
        assert "2" in out


def test_variable_injection() -> None:
    with HaskellInterpreter() as interp:
        out = interp.execute("print (count + 3)", variables={"count": 4})
        assert "7" in out


def test_submit_protocol_returns_final_output() -> None:
    with HaskellInterpreter(output_fields=[{"name": "answer"}]) as interp:
        result = interp.execute("submit (1 :: Int)")
        assert isinstance(result, FinalOutput)
        assert result.output == {"answer": 1}


def test_tool_call_plumbing() -> None:
    def llm_query(prompt: str) -> str:
        return "tool-ok"

    with HaskellInterpreter(tools={"llmQuery": llm_query}) as interp:
        out = interp.execute('llmQuery "ping" >>= putStrLn')
        assert "tool-ok" in out


def test_tool_timeout_surface_in_output() -> None:
    def slow_tool(prompt: str) -> str:
        time.sleep(0.05)
        return "late"

    with HaskellInterpreter(tools={"llmQuery": slow_tool}, tool_call_timeout_s=0.01) as interp:
        out = interp.execute('llmQuery "slow" >>= putStrLn')
        assert "[TIMEOUT]" in out
