"""Both retrieval paths against live stores.

Prerequisites:

    docker compose up -d
    uv run agrirag-seed --reset
    uv run agrirag-index --extract

Skips rather than fails when a store is unavailable, so the unit suite stays
runnable on a laptop with nothing running.
"""

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from agrirag.agent.tools import graph_query, semantic_search
from agrirag.config import get_settings
from agrirag.graph.entities import link_entities
from agrirag.graph.neo4j_store import Neo4jGraphStore
from agrirag.vector.lancedb_store import LanceDBVectorStore, VectorStoreNotBuiltError


@pytest_asyncio.fixture
async def graph() -> AsyncIterator[Neo4jGraphStore]:
    settings = get_settings()
    store = Neo4jGraphStore(
        settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password, settings.neo4j_database
    )
    try:
        await store.verify_connectivity()
    except (ServiceUnavailable, Neo4jError) as exc:
        await store.close()
        pytest.skip(f"Neo4j unavailable: {exc}")
    try:
        yield store
    finally:
        await store.close()


@pytest_asyncio.fixture
async def vectors() -> AsyncIterator[LanceDBVectorStore]:
    store = LanceDBVectorStore()
    try:
        await store.count()
    except VectorStoreNotBuiltError as exc:
        pytest.skip(f"LanceDB index not built: {exc}")
    try:
        yield store
    finally:
        await store.close()


# --------------------------------------------------------------------------
# Entity linking
# --------------------------------------------------------------------------


async def test_entity_linking_resolves_crop_and_region(graph: Neo4jGraphStore) -> None:
    entities = await link_entities(graph, "organic treatments for maize pests in Konya")
    resolved = {e.label: e.id for e in entities}

    assert resolved["Crop"] == "crop_maize"
    assert resolved["Region"] == "reg_konya"


async def test_entity_linking_resolves_turkish_names(graph: Neo4jGraphStore) -> None:
    """Aliases carry the Turkish names, so either language resolves."""
    entities = await link_entities(graph, "sune zararlisi bugday tarlasinda")
    resolved = {e.label: e.id for e in entities}

    assert resolved["Crop"] == "crop_wheat"
    assert resolved["Pest"] == "pest_sunn_pest"


async def test_entity_linking_returns_nothing_for_an_off_domain_question(
    graph: Neo4jGraphStore,
) -> None:
    """A question with no agricultural entity must not hallucinate a match -
    this is what lets the agent abstain rather than answer from noise."""
    assert await link_entities(graph, "How do I configure a Kubernetes ingress?") == []


# --------------------------------------------------------------------------
# Graph retrieval
# --------------------------------------------------------------------------


async def test_graph_query_answers_the_headline_multi_hop_question(
    graph: Neo4jGraphStore,
) -> None:
    result = await graph_query(
        graph,
        "Which organic-approved treatments exist for maize pests in Konya?",
        template_name="treatments_for_crop_pests",
        extra_params={"organic_only": True},
    )

    assert result.found
    assert result.params["crop_id"] == "crop_maize"
    assert result.params["region_id"] == "reg_konya"

    inputs = {row["input_name"] for row in result.rows}
    assert inputs == {"Bacillus thuringiensis", "Spinosad"}
    # Every row must be organic - the filter is the point of the question.
    assert all(row["organic"] is True for row in result.rows)
    # Cotton bollworm attacks maize but is not established in Konya.
    assert "Cotton bollworm" not in {row["pest_name"] for row in result.rows}


async def test_graph_query_reaches_four_hops_for_regulation(graph: Neo4jGraphStore) -> None:
    result = await graph_query(
        graph,
        "Are any maize treatments restricted in Konya?",
        template_name="restricted_inputs_for_crop_in_region",
    )

    assert result.found
    assert "Chlorpyrifos-ethyl" in {row["input_name"] for row in result.rows}
    # Only the Turkish instrument is in force in Konya.
    assert {row["jurisdiction"] for row in result.rows} == {"Turkey"}


async def test_graph_rows_are_linearised_into_sentences(graph: Neo4jGraphStore) -> None:
    """This is what makes DeepEval's contextual metrics applicable to graph facts:
    they arrive in the same shape as text chunks."""
    result = await graph_query(
        graph, "What pests affect cotton?", template_name="pest_profile_for_crop"
    )

    assert result.context
    assert len(result.context) == len(result.rows)
    assert all(isinstance(line, str) and len(line) > 40 for line in result.context)
    assert any("Cotton bollworm" in line for line in result.context)


async def test_graph_query_reports_an_unresolvable_question_instead_of_guessing(
    graph: Neo4jGraphStore,
) -> None:
    result = await graph_query(graph, "How do I configure a Kubernetes ingress?")

    assert not result.found
    assert result.error
    assert result.context == []


async def test_the_executed_cypher_is_returned_for_traceability(graph: Neo4jGraphStore) -> None:
    """The API surfaces this so a reviewer can see exactly what ran."""
    result = await graph_query(
        graph, "What should I rotate with wheat?", template_name="rotation_partners"
    )

    assert "ROTATES_WITH" in result.cypher
    assert "LIMIT" in result.cypher  # the guard injects one


# --------------------------------------------------------------------------
# Semantic retrieval
# --------------------------------------------------------------------------


async def test_semantic_search_finds_the_right_document(vectors: LanceDBVectorStore) -> None:
    result = await semantic_search(
        vectors, "What are the principles behind integrated pest management?", k=3
    )

    assert result.found
    assert result.hits[0].doc_title == "Integrated Pest Management"
    assert result.hits[0].score > 0.6


async def test_semantic_scores_are_similarity_not_distance(
    vectors: LanceDBVectorStore,
) -> None:
    """LanceDB returns cosine distance; leaking that convention would invert
    every threshold comparison in the agent."""
    result = await semantic_search(vectors, "soil salinity and sodicity", k=5)

    scores = [h.score for h in result.hits]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


async def test_chunks_carry_their_heading_into_the_context(
    vectors: LanceDBVectorStore,
) -> None:
    result = await semantic_search(vectors, "why gypsum is used on sodic soils", k=2)

    assert all(line.startswith("[") for line in result.context)
    assert any("Sodic" in line or "sodic" in line for line in result.context)


async def test_the_index_covers_the_whole_corpus(vectors: LanceDBVectorStore) -> None:
    assert await vectors.count() == 107


# --------------------------------------------------------------------------
# Provenance written by the extraction pass
# --------------------------------------------------------------------------


async def test_extraction_did_not_alter_the_deterministic_backbone(
    graph: Neo4jGraphStore,
) -> None:
    """The whole point of MATCH-only linking: the CSV-defined graph is untouched,
    so eval thresholds do not drift when the corpus is re-ingested."""
    rows = await graph.read("MATCH (n) WHERE NOT n:Chunk AND NOT n:Document RETURN count(n) AS c")
    assert rows[0]["c"] == 75


async def test_chunks_link_back_to_graph_entities(graph: Neo4jGraphStore) -> None:
    rows = await graph.read("MATCH (c:Chunk)-[:MENTIONS]->(e) RETURN count(*) AS c")
    assert rows[0]["c"] > 100


async def test_a_graph_fact_can_be_traced_to_explanatory_prose(
    graph: Neo4jGraphStore,
) -> None:
    """Provenance is what makes the combined retrieval route worth having: the
    graph says maize is susceptible to fall armyworm, the corpus says why."""
    rows = await graph.read(
        """
        MATCH (:Crop {id: 'crop_maize'})-[:SUSCEPTIBLE_TO]->(p:Pest {id: 'pest_fall_armyworm'})
        MATCH (c:Chunk)-[:MENTIONS]->(p)
        MATCH (c)-[:PART_OF]->(d:Document)
        RETURN DISTINCT d.title AS title
        """
    )

    assert "Fall Armyworm Management" in {row["title"] for row in rows}


async def test_every_chunk_belongs_to_a_document(graph: Neo4jGraphStore) -> None:
    rows = await graph.read(
        "MATCH (c:Chunk) WHERE NOT (c)-[:PART_OF]->(:Document) RETURN count(c) AS c"
    )
    assert rows[0]["c"] == 0
