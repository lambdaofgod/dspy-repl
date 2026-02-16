"""
Abstract interpreter interface for code execution environments.

This module defines the CodeInterpreter protocol that allows RLM and other
code-executing modules to work with different interpreter implementations.
"""

from typing import Any, Callable, Protocol, runtime_checkable

SIMPLE_TYPES = (str, int, float, bool, list, dict, type(None))


class CodeInterpreterError(RuntimeError):
    """Error raised during code interpretation."""


class FinalOutput:
    """Returned by interpreter.execute() when SUBMIT() is called."""

    def __init__(self, output: Any):
        self.output = output

    def __repr__(self) -> str:
        return f"FinalOutput({self.output!r})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FinalOutput):
            return NotImplemented
        return self.output == other.output


@runtime_checkable
class CodeInterpreter(Protocol):
    """Protocol for code execution environments."""

    @property
    def tools(self) -> dict[str, Callable[..., str]]: ...

    def start(self) -> None: ...

    def execute(self, code: str, variables: dict[str, Any] | None = None) -> Any: ...

    def shutdown(self) -> None: ...


__all__ = ["SIMPLE_TYPES", "CodeInterpreter", "CodeInterpreterError", "FinalOutput"]
