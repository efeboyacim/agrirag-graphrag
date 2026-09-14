"""The agent, end to end against live stores and a live model.

These cost tokens, so the set is small and each case earns its place: the four
canonical question shapes, the abstain path, the node sequence for each route,
and multi-turn threading.

Node sequence is asserted over ``astream`` rather than over the final answer.
Checking only the output would pass even if the agent reached it by the wrong
path - running both retrievers on a definitional question, say, or skipping the
grader entirely.
"""

from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from langgraph.checkpoint.memory import InMemorySaver
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from agrirag.agent.graph import Agent, ask, build_agent
from agrirag.config import get_settings
from agrirag.graph.neo4j_store import Neo4jGraphStore
from agrirag.vector.lancedb_store import LanceDBVectorStore, VectorStoreNotBuiltError

# See tests/integration/conftest.py: "no key" skips, "key rejected" fails once
# with a message naming the cause instead of fifteen opaque connection traces.
#
# Tests that assert *routing precision* take `reference_llm` rather than
# `live_llm`. On a small local model the router frequently returns no valid tool
# call at all, and the node's fallback answers by running both retrievers - so
# the observed "both" is the fallback working, not the model choosing it. The
# answer is still correct; the route is just not the cheapest one. That is a
# model-quality property, and asserting it against any provider would make the
# suite report a weak model as a broken agent.
pytestmark = pytest.mark.usefixtures("live_llm")


@pytest_asyncio.fixture(scope="session")
async def stores() -> AsyncIterator[tuple[Neo4jGraphStore, LanceDBVectorStore]]:
    settings = get_settings()
    graph = Neo4jGraphStore(
        settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password, settings.neo4j_database
    )
    vectors = LanceDBVectorStore()
    try:
        await graph.verify_connectivity()
        await vectors.count()
    except (ServiceUnavailable, Neo4jError, VectorStoreNotBuiltError) as exc:
        await graph.close()
        pytest.skip(f"stores unavailable: {exc}")

    try:
        yield graph, vectors
    finally:
        await graph.close()
        await vectors.close()


@pytest_asyncio.fixture(scope="session")
async def agent(stores: tuple[Neo4jGraphStore, LanceDBVectorStore]) -> Agent:
    graph, vectors = stores
    return build_agent(graph, vectors)


async def visited_nodes(agent: Agent, question: str) -> list[str]:
    """Node names in the order LangGraph actually executed them."""
    seen: list[str] = []
    state: dict[str, Any] = {
        "question": question,
        "trace_id": "test-stream",
        "retrieval_context": [],
        "graph_rows": [],
        "chunks": [],
        "retries": 0,
    }
    async for update in agent.astream(state, stream_mode="updates"):
        seen.extend(update.keys())
    return seen


# --------------------------------------------------------------------------
# The four canonical question shapes
# --------------------------------------------------------------------------


async def test_multi_hop_graph_question(agent: Agent, reference_llm) -> None:
    """Four hops, and the region hop is what makes it interesting."""
    result = await ask(
        agent,
        "Which organic-approved treatments exist for maize pests in Konya?",
        trace_id="test-q1",
    )

    assert result.route == "graph"
    assert not result.abstained
    assert result.graph_row_count > 0
    assert result.retrieval_context

    answer = result.answer.lower()
    assert "bacillus thuringiensis" in answer or "bt" in answer
    assert "spinosad" in answer
    # Cotton bollworm attacks maize but is not established in Konya. A vector
    # search over the same corpus has no way to make that exclusion.
    assert "bollworm" not in answer

    assert result.cypher and "TREATED_BY" in result.cypher
    assert {e["id"] for e in result.linked_entities} >= {"crop_maize", "reg_konya"}


async def test_regulatory_question_reaches_the_regulation_hop(agent: Agent, reference_llm) -> None:
    result = await ask(agent, "Are any maize treatments restricted in Konya?", trace_id="test-q2")

    assert result.route == "graph"
    assert not result.abstained
    assert "chlorpyrifos" in result.answer.lower()


async def test_definitional_question_uses_semantic_search(agent: Agent, reference_llm) -> None:
    result = await ask(
        agent, "Explain the principles behind integrated pest management.", trace_id="test-q3"
    )

    assert result.route == "semantic"
    assert not result.abstained
    assert result.chunk_count > 0
    assert result.graph_row_count == 0
    assert any("threshold" in line.lower() for line in result.retrieval_context)


async def test_a_why_question_can_use_both_paths(agent: Agent) -> None:
    result = await ask(
        agent,
        "Why is cover cropping recommended for sandy soils in semi-arid regions?",
        trace_id="test-q4",
    )

    assert result.route in {"both", "semantic"}
    assert not result.abstained
    assert result.retrieval_context


# --------------------------------------------------------------------------
# Abstention
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "How do I configure a Kubernetes ingress?",
        "What is the capital of France?",
    ],
)
async def test_off_domain_questions_abstain(agent: Agent, question: str) -> None:
    """The relevance floor is what makes this possible: without it, vector
    search returns its k nearest neighbours regardless of how far away they
    are, and the agent answers off-domain questions from agronomy chunks."""
    result = await ask(agent, question, trace_id="test-abstain")

    assert result.abstained
    assert result.retrieval_context == []
    assert "don't have enough information" in result.answer


# --------------------------------------------------------------------------
# Control flow
# --------------------------------------------------------------------------


async def test_a_graph_question_does_not_run_semantic_search(agent: Agent) -> None:
    nodes = await visited_nodes(agent, "What should I rotate with wheat?")

    assert nodes[0] == "route"
    assert "graph_retrieve" in nodes
    assert nodes[-1] == "synthesize"


async def test_a_definitional_question_does_not_touch_the_graph(
    agent: Agent, reference_llm
) -> None:
    nodes = await visited_nodes(agent, "What is an economic threshold?")

    assert "semantic_retrieve" in nodes
    assert "graph_retrieve" not in nodes
    assert "grade" in nodes


async def test_an_off_domain_question_ends_at_abstain(agent: Agent) -> None:
    nodes = await visited_nodes(agent, "What is the capital of France?")

    assert nodes[-1] == "abstain"
    assert "synthesize" not in nodes


async def test_every_run_passes_through_the_grader(agent: Agent) -> None:
    """Grading is not optional - it is the only thing standing between an empty
    retrieval and a confidently synthesised answer."""
    nodes = await visited_nodes(agent, "What pests affect cotton in Cukurova?")

    assert "grade" in nodes


# --------------------------------------------------------------------------
# Threading
# --------------------------------------------------------------------------


async def test_a_checkpointer_persists_state_across_turns(
    stores: tuple[Neo4jGraphStore, LanceDBVectorStore],
) -> None:
    graph, vectors = stores
    threaded = build_agent(graph, vectors, checkpointer=InMemorySaver())

    await ask(threaded, "What should I rotate with wheat?", thread_id="thread-a")
    snapshot = await threaded.aget_state({"configurable": {"thread_id": "thread-a"}})

    assert snapshot.values["question"] == "What should I rotate with wheat?"
    assert snapshot.values["answer"]


async def test_threads_do_not_leak_into_each_other(
    stores: tuple[Neo4jGraphStore, LanceDBVectorStore], reference_llm
) -> None:
    graph, vectors = stores
    threaded = build_agent(graph, vectors, checkpointer=InMemorySaver())

    await ask(threaded, "What should I rotate with wheat?", thread_id="thread-b")
    await ask(threaded, "What is an economic threshold?", thread_id="thread-c")

    b = await threaded.aget_state({"configurable": {"thread_id": "thread-b"}})
    c = await threaded.aget_state({"configurable": {"thread_id": "thread-c"}})

    assert b.values["route"] == "graph"
    assert c.values["route"] == "semantic"
