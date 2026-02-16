from __future__ import annotations

import time

import pytest

from dspy_repl.core.code_interpreter import CodeInterpreterError, FinalOutput
from dspy_repl.interpreters.sql_interpreter import SQLInterpreter


def test_scalar_and_list_variable_materialization() -> None:
    interp = SQLInterpreter()
    try:
        out = interp.execute(
            "SELECT (SELECT value FROM question) AS q, (SELECT COUNT(*) FROM items) AS n;",
            variables={"question": "hello", "items": [1, 2, 3]},
        )
        assert "hello" in out
        assert "3" in out
    finally:
        interp.shutdown()


def test_dict_of_dicts_materializes_columns() -> None:
    interp = SQLInterpreter()
    try:
        out = interp.execute(
            "SELECT city FROM users WHERE key='alice';",
            variables={"users": {"alice": {"city": "NYC"}, "bob": {"city": "SF"}}},
        )
        assert "NYC" in out
    finally:
        interp.shutdown()


def test_submit_returns_final_output() -> None:
    interp = SQLInterpreter()
    try:
        result = interp.execute("SELECT SUBMIT(json_object('answer', '42'));")
        assert isinstance(result, FinalOutput)
        assert result.output == {"answer": "42"}
    finally:
        interp.shutdown()


def test_timeout_raises_code_interpreter_error() -> None:
    interp = SQLInterpreter(statement_timeout_seconds=0.001)
    try:
        with pytest.raises(CodeInterpreterError, match="timed out"):
            interp.execute("WITH RECURSIVE t(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM t) SELECT n FROM t;")
    finally:
        interp.shutdown()


def test_tool_timeout_path() -> None:
    def slow_tool(*args: object) -> str:
        time.sleep(0.03)
        return "ok"

    interp = SQLInterpreter(tools={"slow_tool": slow_tool}, tool_timeout_seconds=0.01)
    try:
        with pytest.raises(CodeInterpreterError, match="SQLite error"):
            interp.execute("SELECT slow_tool('x');")
    finally:
        interp.shutdown()


def test_profile_counters_update() -> None:
    interp = SQLInterpreter()
    try:
        interp.execute("SELECT 1;")
        profile = interp.get_profile()
        assert profile["execute_calls"] >= 1
        assert profile["query_exec_seconds"] >= 0.0
    finally:
        interp.shutdown()


def test_split_sql_statements_keeps_semicolons_inside_strings() -> None:
    interp = SQLInterpreter()
    try:
        out = interp.execute("SELECT 'a; b' AS value;")
        assert "a; b" in out
    finally:
        interp.shutdown()


def test_register_tools_refreshes_updated_callable() -> None:
    state = {"value": "first"}

    def dynamic_tool(*_args: object) -> str:
        return state["value"]

    interp = SQLInterpreter(tools={"dynamic_tool": dynamic_tool})
    try:
        first = interp.execute("SELECT dynamic_tool('x');")
        assert "first" in first

        state["value"] = "second"
        second = interp.execute("SELECT dynamic_tool('x');")
        assert "second" in second
    finally:
        interp.shutdown()
