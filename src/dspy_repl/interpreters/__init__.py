from typing import Any

__all__ = ["load_haskell_interpreter", "load_scheme_interpreter", "load_sql_interpreter"]


def load_scheme_interpreter() -> Any:
    from dspy_repl.interpreters.scheme_interpreter import SchemeInterpreter

    return SchemeInterpreter


def load_sql_interpreter() -> Any:
    from dspy_repl.interpreters.sql_interpreter import SQLInterpreter

    return SQLInterpreter


def load_haskell_interpreter() -> Any:
    from dspy_repl.interpreters.haskell_interpreter import HaskellInterpreter

    return HaskellInterpreter
