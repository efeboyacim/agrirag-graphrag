"""End-to-end against the running container.

Unlike the rest of the suite these go over HTTP to the deployed stack, so they
exercise the Docker wiring itself - the bind mounts, the container-internal
hostnames, whether the portkey configs and the corpus made it into the image.
Several real defects in this project were only visible from outside the
container and invisible to in-process tests.

Tests that need a working model take the ``live_llm`` fixture, which separates
"no key configured" (skip) from "key rejected" (one clear failure). Health, schema
and validation tests deliberately do not take it - they must keep working without
a credential, and they are what proves the container wiring independently of the
LLM.

Prerequisites:

    docker compose up -d
    uv run agrirag-seed --reset
    uv run agrirag-index --extract
"""

import httpx
import pytest

from agrirag.config import get_settings

BASE_URL = "http://localhost:8000"

#: A local model on CPU takes minutes per question where a hosted one takes
#: seconds. Two structural tests failed purely on a 120s read timeout when the
#: provider was switched - a timeout is not a defect in the system under test.
TIMEOUT = 600.0 if get_settings().llm_provider == "ollama" else 120.0


def _reachable() -> bool:
    try:
        return httpx.get(f"{BASE_URL}/health", timeout=5).status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _reachable(), reason=f"no API at {BASE_URL} - run: docker compose up -d"
)


@pytest.fixture
async def client():
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=TIMEOUT) as c:
        yield c


# --------------------------------------------------------------------------
# Health and schema
# --------------------------------------------------------------------------


async def test_every_dependency_is_reachable_from_inside_the_container(client) -> None:
    """The container resolves neo4j and gateway by compose hostname, and finds
    the LanceDB index on its bind mount. Each of those has broken at least once."""
    body = (await client.get("/health")).json()

    assert body["status"] == "ok", body
    assert body["neo4j"] == "ok"
    assert body["lancedb"] == "ok"
    assert body["gateway"] == "ok"
    assert body["chunks"] == 107


async def test_graph_schema_reports_the_seeded_counts(client) -> None:
    body = (await client.get("/api/v1/graph/schema")).json()

    counts = {item["label"]: item["count"] for item in body["labels"]}
    assert counts["Crop"] == 10
    assert counts["Pest"] == 16
    assert counts["Regulation"] == 5

    rels = {item["type"]: item["count"] for item in body["relationships"]}
    assert rels["TREATED_BY"] == 27
    assert rels["SUSCEPTIBLE_TO"] == 21


async def test_the_openapi_document_is_served(client) -> None:
    """Swagger is the demo surface; a broken schema breaks the demo."""
    schema = (await client.get("/openapi.json")).json()

    assert "/api/v1/ask" in schema["paths"]
    assert "/api/v1/ingest" in schema["paths"]
    assert "/api/v1/graph/schema" in schema["paths"]


# --------------------------------------------------------------------------
# Asking
# --------------------------------------------------------------------------


async def test_a_multi_hop_question_is_answered_from_the_graph(client, reference_llm) -> None:
    response = await client.post(
        "/api/v1/ask",
        json={"question": "Which organic-approved treatments exist for maize pests in Konya?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "graph"
    assert body["graph_row_count"] > 0
    assert not body["abstained"]
    assert "spinosad" in body["answer"].lower()
    assert "TREATED_BY" in body["cypher"]


async def test_the_graph_returns_rotation_partners_the_corpus_never_lists(
    client, reference_llm
) -> None:
    """The clearest demonstration of what the graph adds.

    ROTATES_WITH edges exist only in the seed CSVs - the prose discusses the
    wheat-chickpea pairing at length and barely mentions the others. So the
    graph path returns several partners where semantic search returns roughly
    one, on the same question.

    Note the honest framing: for questions whose answer happens to sit inside a
    single document, semantic search does nearly as well. The graph wins where
    the answer is a set of relationships that no single document enumerates.
    """
    payload = {"question": "What should I rotate with wheat, and why?"}

    with_graph = (await client.post("/api/v1/ask", json=payload)).json()
    semantic_only = (
        await client.post("/api/v1/ask", json={**payload, "force_route": "semantic"})
    ).json()

    assert with_graph["graph_row_count"] >= 4
    assert semantic_only["graph_row_count"] == 0
    assert semantic_only["route"] == "semantic"

    # The graph path names partners the prose does not enumerate.
    answer = with_graph["answer"].lower()
    assert sum(crop in answer for crop in ("chickpea", "sunflower", "sugar beet", "cotton")) >= 3


async def test_a_definitional_question_uses_the_corpus(client, live_llm) -> None:
    body = (
        await client.post("/api/v1/ask", json={"question": "What is an economic threshold?"})
    ).json()

    assert body["route"] == "semantic"
    assert body["chunk_count"] > 0
    assert body["graph_row_count"] == 0


async def test_an_off_domain_question_abstains(client, live_llm) -> None:
    body = (
        await client.post(
            "/api/v1/ask", json={"question": "How do I configure a Kubernetes ingress?"}
        )
    ).json()

    assert body["abstained"] is True
    assert body["graph_row_count"] == 0
    assert body["chunk_count"] == 0


async def test_a_partly_answerable_question_answers_and_says_what_is_missing(
    client, reference_llm
) -> None:
    """A compound question where the graph holds half the answer. Abstaining
    while holding verified rows would be worse than answering with a caveat -
    which is the branch this pins."""
    body = (
        await client.post(
            "/api/v1/ask",
            json={
                "question": (
                    "Which organic-approved treatments exist for maize pests in Konya, "
                    "and are any of them restricted there?"
                )
            },
        )
    ).json()

    assert not body["abstained"]
    assert body["graph_row_count"] > 0
    assert "spinosad" in body["answer"].lower()


async def test_the_request_id_round_trips(client, live_llm) -> None:
    response = await client.post(
        "/api/v1/ask",
        json={"question": "What pests affect cotton?"},
        headers={"x-request-id": "e2e-trace-1"},
    )

    assert response.headers["x-request-id"] == "e2e-trace-1"
    assert response.json()["request_id"] == "e2e-trace-1"


async def test_a_malformed_request_is_rejected_before_reaching_the_agent(client) -> None:
    response = await client.post("/api/v1/ask", json={"question": "no"})

    assert response.status_code == 422


# --------------------------------------------------------------------------
# Ingest
# --------------------------------------------------------------------------


async def test_rebuilding_the_vector_index_leaves_search_working(client, live_llm) -> None:
    """The store caches an open table handle. Rebuilding drops that table, so a
    stale handle would make every subsequent search fail."""
    ingest = await client.post(
        "/api/v1/ingest",
        json={"rebuild_vectors": True, "extract_entities": False},
        timeout=300.0,
    )

    assert ingest.status_code == 200
    body = ingest.json()
    assert body["vectors_rebuilt"] is True
    assert body["chunks"] == 107
    assert body["extraction_ran"] is False

    after = (
        await client.post("/api/v1/ask", json={"question": "How does soil solarization work?"})
    ).json()
    assert after["chunk_count"] > 0
    assert not after["abstained"]
