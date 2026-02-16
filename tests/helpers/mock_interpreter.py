from __future__ import annotations

from typing import Any


class MockInterpreter:
    """Scripted interpreter responses for deterministic RLM tests."""

    def __init__(self, responses: list[Any] | None = None):
        self.responses = list(responses or [])
        self.tools: dict[str, Any] = {}
        self.output_fields: list[dict[str, Any]] | None = None
        self._idx = 0
        self.call_history: list[tuple[str, dict[str, Any] | None]] = []

    def start(self) -> None:
        return None

    def shutdown(self) -> None:
        return None

    def execute(self, code: str, variables: dict[str, Any] | None = None) -> Any:
        self.call_history.append((code, variables))
        if not self.responses:
            return None
        item = self.responses[self._idx % len(self.responses)]
        self._idx += 1
        if isinstance(item, Exception):
            raise item
        return item
