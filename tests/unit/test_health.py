"""Health endpoint behaviour."""

from tests.conftest import FakeGraphStore


async def test_health_reports_ok_when_graph_reachable(client_factory) -> None:
    async with client_factory(FakeGraphStore(healthy=True)) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == "0.1.0"
    assert body["neo4j"] == "ok"
    assert body["lancedb"] == "ok"
    assert body["chunks"] == 107


async def test_health_reports_degraded_when_graph_unreachable(client_factory) -> None:
    """A dead dependency degrades the payload but must not 500 the endpoint."""
    async with client_factory(FakeGraphStore(healthy=False)) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "degraded"
    assert body["neo4j"] == "unavailable"
    # A dead graph store must not be reported as a dead vector store.
    assert body["lancedb"] == "ok"


async def test_health_is_also_mounted_under_the_versioned_prefix(client_factory) -> None:
    async with client_factory(FakeGraphStore()) as client:
        response = await client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


async def test_request_id_is_echoed_back(client_factory) -> None:
    async with client_factory(FakeGraphStore()) as client:
        response = await client.get("/health", headers={"x-request-id": "trace-abc"})

    assert response.headers["x-request-id"] == "trace-abc"


async def test_request_id_is_generated_when_absent(client_factory) -> None:
    async with client_factory(FakeGraphStore()) as client:
        response = await client.get("/health")

    assert response.headers.get("x-request-id")
