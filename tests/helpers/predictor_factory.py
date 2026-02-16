from __future__ import annotations

from typing import Any

from dspy_repl.compat import Prediction


def make_mock_predictor(responses: list[dict[str, Any]]):
    """Factory for deterministic predictor outputs."""

    class MockPredictor:
        def __init__(self):
            self.idx = 0

        def _next_response(self) -> Prediction:
            payload = responses[self.idx % len(responses)]
            self.idx += 1
            return Prediction(**payload)

        def __call__(self, **kwargs: Any) -> Prediction:
            return self._next_response()

        async def acall(self, **kwargs: Any) -> Prediction:
            return self._next_response()

    return MockPredictor()
