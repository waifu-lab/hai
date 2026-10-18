import json
from typing import Callable, List

import httpx
import pytest


class MockHTTP:
    """Routes every AsyncClient created by hai.providers to *handler* and records the requests."""

    def __init__(self) -> None:
        self.requests: List[httpx.Request] = []
        self.handler: Callable[[httpx.Request], httpx.Response] = lambda _req: httpx.Response(500)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.handler(request)

    def bodies(self) -> List[dict]:
        return [json.loads(r.content) for r in self.requests]


@pytest.fixture
def mock_http(monkeypatch):
    mock = MockHTTP()
    real_client = httpx.AsyncClient

    def client_factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(mock)
        return real_client(*args, **kwargs)

    for module in (
        "hai.providers.providers",
        "hai.providers.openai",
        "hai.providers.anthropic",
        "hai.providers.google",
    ):
        monkeypatch.setattr(f"{module}.AsyncClient", client_factory)
    # Retries sleep with exponential backoff; skip the wait in tests
    monkeypatch.setattr("hai.providers.providers._retry_delay", lambda _attempt, _exc: 0.0)
    return mock
