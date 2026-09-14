"""API surface: validation, wiring, and error translation.

The agent is stubbed here. These tests are about the HTTP layer - does a bad
request get rejected, does an infrastructure failure become an actionable reply,
does the request id reach both the agent and the response. The agent's own
behaviour is covered by the integration suite.
"""

from typing import Any

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from neo4j.exceptions import ServiceUnavailable

from agrirag.agent.state import AgentResult, Citation
from agrirag.api.deps import get_agent
from agrirag.llm.portkey_client import LLMConfigurationError
from agrirag.vector.lancedb_store import VectorStoreNotBuiltError
from tests.conftest import FakeGraphStore, FakeVectorStore


class FakeAgent:
    """Records what it was asked, returns a canned result."""

    def __init__(self, result: AgentResult | None = None, raises: Exception | None = None) -> None:
        self.result = result
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    async def ainvoke(self, state: dict[str, Any], config: dict[str, Any] | None = None) -> Any:
        self.calls.append({"state": state, "config": config or {}})
        if self.raises:
            raise self.raises
        result = self.result or _result()
        return {
            "question": state["question"],
            "trace_id": state.get("trace_id", ""),
            "answer": result.answer,
            "route": result.route,
            "route_reason": result.route_reason,
            "retrieval_context": result.retrieval_context,
            "citations": [c.model_dump() for c in result.citations],
            "abstained": result.abstained,
            "template": result.template,
            "cypher": result.cypher,
            "linked_entities": result.linked_entities,
            "graph_rows": [{"x": 1}] * result.graph_row_count,
            "chunks": [{"chunk_id": "c"}] * result.chunk_count,
            "retries": result.retries,
        }


def _result(**overrides: Any) -> AgentResult:
    base: dict[str, Any] = {
        "question": "q",
        "answer": "Bacillus thuringiensis and spinosad.",
        "route": "graph",
        "route_reason": "relational lookup",
        "retrieval_context": ["Maize is susceptible to fall armyworm."],
        "citations": [Citation(kind="graph", ref="crop_maize", label="Maize")],
        "template": "treatments_for_crop_pests",
        "cypher": "MATCH (c:Crop) RETURN c LIMIT 25",
        "linked_entities": [{"id": "crop_maize", "label": "Crop"}],
        "graph_row_count": 3,
    }
    base.update(overrides)
    return AgentResult(**base)


@pytest.fixture
def api(app: FastAPI):
    """Client factory with the agent dependency overridden."""

    def _make(agent: FakeAgent) -> AsyncClient:
        app.state.graph_store = FakeGraphStore()
        app.state.vector_store = FakeVectorStore()
        app.dependency_overrides[get_agent] = lambda: agent
        return AsyncClient(
            # Starlette's error middleware calls the 500 handler and *then*
            # re-raises so the server can log the traceback. Under uvicorn the
            # client still receives the handler's response; the ASGI transport
            # would instead surface the exception to the test, hiding what a
            # real caller actually gets.
            transport=ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        )

    yield _make
    app.dependency_overrides.clear()


# --------------------------------------------------------------------------
# /ask
# --------------------------------------------------------------------------


async def test_ask_returns_the_answer_and_the_audit_trail(api) -> None:
    """The response exposes how the answer was reached, not just the answer -
    that is what lets a reviewer check grounding instead of trusting it."""
    async with api(FakeAgent()) as client:
        response = await client.post("/api/v1/ask", json={"question": "maize pests in Konya?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("Bacillus")
    assert body["route"] == "graph"
    assert body["route_reason"]
    assert body["retrieval_context"]
    assert body["cypher"]
    assert body["template"] == "treatments_for_crop_pests"
    assert body["graph_row_count"] == 3
    assert body["citations"][0]["ref"] == "crop_maize"


async def test_the_request_id_reaches_the_agent_and_the_response(api) -> None:
    """It is also the Portkey trace id, so a reported request id maps to a trace."""
    agent = FakeAgent()
    async with api(agent) as client:
        response = await client.post(
            "/api/v1/ask",
            json={"question": "what rotates with wheat?"},
            headers={"x-request-id": "trace-xyz"},
        )

    assert response.json()["request_id"] == "trace-xyz"
    assert response.headers["x-request-id"] == "trace-xyz"
    assert agent.calls[0]["state"]["trace_id"] == "trace-xyz"


async def test_thread_id_is_passed_through_to_the_checkpointer(api) -> None:
    agent = FakeAgent()
    async with api(agent) as client:
        await client.post("/api/v1/ask", json={"question": "a question", "thread_id": "conv-1"})

    assert agent.calls[0]["config"]["configurable"]["thread_id"] == "conv-1"


async def test_force_route_skips_the_router(api) -> None:
    """The comparison affordance: same question, no graph, visibly worse answer."""
    agent = FakeAgent()
    async with api(agent) as client:
        await client.post(
            "/api/v1/ask",
            json={"question": "maize pests in Konya?", "force_route": "semantic"},
        )

    assert agent.calls[0]["state"]["forced_route"] == "semantic"


async def test_an_abstention_is_reported_as_such_not_as_an_error(api) -> None:
    """Declining is correct behaviour, so it is a 200 with a flag - not a 404."""
    agent = FakeAgent(
        _result(
            answer="I don't have enough information to answer that.",
            abstained=True,
            retrieval_context=[],
            citations=[],
            graph_row_count=0,
        )
    )
    async with api(agent) as client:
        response = await client.post("/api/v1/ask", json={"question": "capital of France?"})

    assert response.status_code == 200
    assert response.json()["abstained"] is True


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "ab"},
        {"question": "x" * 1001},
        {"question": "valid question", "force_route": "telepathy"},
    ],
)
async def test_bad_requests_are_rejected_with_422(api, payload: dict[str, Any]) -> None:
    async with api(FakeAgent()) as client:
        response = await client.post("/api/v1/ask", json=payload)

    assert response.status_code == 422


# --------------------------------------------------------------------------
# Error translation
# --------------------------------------------------------------------------


async def test_a_missing_api_key_is_a_503_with_an_actionable_message(api) -> None:
    agent = FakeAgent(raises=LLMConfigurationError("ANTHROPIC_API_KEY is not set. Add it to .env"))
    async with api(agent) as client:
        response = await client.post("/api/v1/ask", json={"question": "a question"})

    assert response.status_code == 503
    body = response.json()
    assert body["error_type"] == "llm_misconfigured"
    assert "ANTHROPIC_API_KEY" in body["detail"]
    assert body["request_id"]


async def test_a_dead_graph_store_is_a_503(api) -> None:
    agent = FakeAgent(raises=ServiceUnavailable("cannot reach bolt://neo4j:7687"))
    async with api(agent) as client:
        response = await client.post("/api/v1/ask", json={"question": "a question"})

    assert response.status_code == 503
    assert response.json()["error_type"] == "graph_unavailable"


async def test_a_missing_vector_index_tells_the_caller_the_command_to_run(api) -> None:
    agent = FakeAgent(raises=VectorStoreNotBuiltError("table missing"))
    async with api(agent) as client:
        response = await client.post("/api/v1/ask", json={"question": "a question"})

    assert response.status_code == 503
    assert "agrirag-index" in response.json()["detail"]


async def test_an_unexpected_error_does_not_leak_internals(api) -> None:
    """The message may contain paths, queries or credentials. The request id is
    how it gets tied back to the full traceback in the logs instead."""
    agent = FakeAgent(raises=RuntimeError("secret connection string postgres://user:pw@host"))
    async with api(agent) as client:
        response = await client.post("/api/v1/ask", json={"question": "a question"})

    assert response.status_code == 500
    body = response.json()
    assert "secret" not in body["detail"]
    assert "postgres" not in body["detail"]
    assert body["error_type"] == "internal_error"
    assert body["request_id"]


# --------------------------------------------------------------------------
# /graph/schema
# --------------------------------------------------------------------------


async def test_graph_schema_lists_the_declared_labels(api) -> None:
    async with api(FakeAgent()) as client:
        response = await client.get("/api/v1/graph/schema")

    assert response.status_code == 200
    body = response.json()
    labels = {item["label"] for item in body["labels"]}
    assert {"Crop", "Pest", "Input", "Regulation"} <= labels
    assert {item["type"] for item in body["relationships"]} >= {"TREATED_BY", "GROWN_IN"}


async def test_graph_schema_exposes_the_exact_prompt_string(api) -> None:
    """What the model sees should be inspectable, not implied."""
    async with api(FakeAgent()) as client:
        body = (await client.get("/api/v1/graph/schema")).json()

    assert "NODE LABELS" in body["prompt_schema"]
    assert "[:SUSCEPTIBLE_TO" in body["prompt_schema"]


# --------------------------------------------------------------------------
# OpenAPI
# --------------------------------------------------------------------------


async def test_every_route_is_versioned_except_the_container_healthcheck(api) -> None:
    async with api(FakeAgent()) as client:
        paths = (await client.get("/openapi.json")).json()["paths"]

    unversioned = {p for p in paths if not p.startswith("/api/v1")}
    assert unversioned == {"/health"}


async def test_the_ask_endpoint_ships_usable_examples(api) -> None:
    """Swagger is the demo surface, so the examples have to be real questions."""
    async with api(FakeAgent()) as client:
        schema = (await client.get("/openapi.json")).json()

    examples = schema["components"]["schemas"]["AskRequest"]["examples"]
    assert len(examples) >= 3
    assert any("Konya" in ex["question"] for ex in examples)
    assert any(ex.get("force_route") == "semantic" for ex in examples)
