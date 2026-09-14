"""Shared test fixtures."""

from collections.abc import Callable
from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from agrirag.main import create_app
from agrirag.vector.store import SearchHit


class FakeGraphStore:
    """In-memory stand-in for Neo4j.

    Lets the unit suite exercise routes without a container. ``healthy=False``
    reproduces an unreachable graph store.
    """

    def __init__(self, *, healthy: bool = True) -> None:
        self.healthy = healthy
        self.queries: list[tuple[str, dict[str, Any]]] = []

    async def verify_connectivity(self) -> None:
        if not self.healthy:
            raise ConnectionError("graph store unreachable")

    async def read(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        self.queries.append((cypher, params or {}))
        return []

    async def close(self) -> None:
        return None


class FakeVectorStore:
    """In-memory stand-in for LanceDB."""

    def __init__(self, *, healthy: bool = True, hits: list[SearchHit] | None = None) -> None:
        self.healthy = healthy
        self._hits = hits or []
        self.queries: list[str] = []

    async def search(self, query: str, k: int = 5) -> list[SearchHit]:
        if not self.healthy:
            raise RuntimeError("vector store unavailable")
        self.queries.append(query)
        return self._hits[:k]

    async def count(self) -> int:
        if not self.healthy:
            raise RuntimeError("vector store unavailable")
        return 107

    async def close(self) -> None:
        return None


@pytest.fixture(autouse=True)
def stub_gateway_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the unit suite off the network.

    The health endpoint probes the Portkey gateway over HTTP. Without this the
    unit tests would either hit a real container (making them depend on Docker)
    or spend three seconds timing out on every call.
    """
    monkeypatch.setattr(
        "agrirag.api.v1.routes_health._gateway_status",
        _ok_gateway,
        raising=True,
    )


async def _ok_gateway() -> str:
    return "ok"


@pytest.fixture
def app() -> FastAPI:
    """App instance with lifespan bypassed - tests inject their own stores."""
    return create_app()


@pytest.fixture
def client_factory(app: FastAPI) -> Callable[..., AsyncClient]:
    """Build an AsyncClient bound to the given stores."""

    def _factory(
        store: FakeGraphStore,
        vectors: FakeVectorStore | None = None,
    ) -> AsyncClient:
        app.state.graph_store = store
        app.state.vector_store = vectors or FakeVectorStore()
        app.state.settings = None
        return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    return _factory
