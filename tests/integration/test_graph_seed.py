"""Integration tests against a seeded Neo4j.

Run against the docker-compose stack:

    docker compose up -d
    uv run agrirag-seed --reset
    uv run pytest tests/integration

These are the tests that would catch a silently broken ingestion: a renamed id, a
dropped CSV row, or an edge whose endpoints stopped matching. They also pin the
two headline multi-hop queries, which are the whole reason this project uses a
graph instead of flat vector search.
"""

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from agrirag.config import get_settings
from agrirag.graph.schema import ALL_LABELS, ALL_REL_TYPES
from agrirag.ingestion.seed import SEED_DIR
from agrirag.ingestion.writer import Neo4jWriter

EXPECTED_NODE_COUNTS = {
    "Crop": 10,
    "Region": 5,
    "ClimateZone": 4,
    "SoilType": 6,
    "Pest": 16,
    "Input": 17,
    "Practice": 12,
    "Regulation": 5,
}

EXPECTED_REL_COUNTS = {
    "GROWN_IN": 19,
    "HAS_CLIMATE": 5,
    "HAS_SOIL": 9,
    "SUITED_TO": 17,
    "SUSCEPTIBLE_TO": 21,
    "TREATED_BY": 27,
    "PREVALENT_IN": 19,
    "RECOMMENDED_FOR": 19,
    "MITIGATES": 15,
    "RESTRICTED_BY": 13,
    "APPLIES_IN": 19,
    "ROTATES_WITH": 7,
}


@pytest_asyncio.fixture
async def writer() -> AsyncIterator[Neo4jWriter]:
    """Connect to Neo4j, or skip the whole module if it is not running."""
    settings = get_settings()
    client = Neo4jWriter(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    )
    try:
        await client.__aenter__()
    except (ServiceUnavailable, Neo4jError) as exc:
        await client.close()
        pytest.skip(f"Neo4j not reachable at {settings.neo4j_uri}: {exc}")

    try:
        yield client
    finally:
        await client.close()


async def test_node_counts_match_the_seed_files(writer: Neo4jWriter) -> None:
    counts = await writer.count_nodes()
    domain_counts = {label: counts.get(label, 0) for label in ALL_LABELS}
    assert domain_counts == EXPECTED_NODE_COUNTS


async def test_relationship_counts_match_the_seed_files(writer: Neo4jWriter) -> None:
    counts = await writer.count_relationships()
    domain_counts = {rel: counts.get(rel, 0) for rel in ALL_REL_TYPES}
    assert domain_counts == EXPECTED_REL_COUNTS


async def test_every_edge_in_the_seed_files_exists_in_the_graph(writer: Neo4jWriter) -> None:
    """MATCH-based edge loading fails silently on a bad id - this catches that."""
    from agrirag.ingestion.graph_builder import verify_edges

    assert await verify_edges(writer, SEED_DIR) == []


async def test_uniqueness_constraints_exist(writer: Neo4jWriter) -> None:
    rows = await writer.read("SHOW CONSTRAINTS YIELD labelsOrTypes RETURN labelsOrTypes")
    constrained = {label for row in rows for label in (row["labelsOrTypes"] or [])}
    assert set(ALL_LABELS) <= constrained


async def test_fulltext_index_resolves_a_turkish_alias(writer: Neo4jWriter) -> None:
    """Entity linking depends on this index matching aliases, not just names."""
    rows = await writer.read(
        "CALL db.index.fulltext.queryNodes($index, $term) YIELD node, score "
        "RETURN node.id AS id ORDER BY score DESC LIMIT 3",
        {"index": "entity_names", "term": "bugday"},
    )
    assert rows, "fulltext index returned nothing for the Turkish alias 'bugday'"
    assert rows[0]["id"] == "crop_wheat"


# --------------------------------------------------------------------------
# Headline multi-hop queries. Flat vector search cannot answer either of these.
# --------------------------------------------------------------------------


async def test_demo_q1_organic_treatments_for_maize_pests_in_konya(writer: Neo4jWriter) -> None:
    """Crop -> Pest -> Input, intersected with pests actually present in the region.

    "For maize grown in Konya, which organic-approved treatments exist for its
    major pests?"
    """
    rows = await writer.read(
        """
        MATCH (c:Crop {id: $crop})-[:GROWN_IN]->(r:Region {id: $region})
        MATCH (c)-[:SUSCEPTIBLE_TO]->(p:Pest)-[:PREVALENT_IN]->(r)
        MATCH (p)-[t:TREATED_BY]->(i:Input)
        WHERE i.organic_approved = true
        RETURN p.id AS pest, i.id AS input, t.efficacy AS efficacy, t.phi_days AS phi
        ORDER BY pest, input
        """,
        {"crop": "crop_maize", "region": "reg_konya"},
    )

    pairs = {(row["pest"], row["input"]) for row in rows}
    assert pairs == {
        ("pest_corn_borer", "inp_bt"),
        ("pest_fall_armyworm", "inp_bt"),
        ("pest_fall_armyworm", "inp_spinosad"),
    }
    # Cotton bollworm attacks maize but is not established in Konya - the region
    # hop is what excludes it, and that is exactly the hop vector search lacks.
    assert all(row["pest"] != "pest_cotton_bollworm" for row in rows)


async def test_demo_q1b_restricted_inputs_reach_four_hops(writer: Neo4jWriter) -> None:
    """Crop -> Pest -> Input -> Regulation -> Region: the full four-hop path.

    "...and are any of them restricted there?"
    """
    rows = await writer.read(
        """
        MATCH (c:Crop {id: $crop})-[:SUSCEPTIBLE_TO]->(p:Pest)-[:TREATED_BY]->(i:Input)
        MATCH (i)-[:RESTRICTED_BY]->(reg:Regulation)-[:APPLIES_IN]->(:Region {id: $region})
        RETURN DISTINCT i.id AS input, reg.id AS regulation
        ORDER BY input, regulation
        """,
        {"crop": "crop_maize", "region": "reg_konya"},
    )

    restricted = {row["input"] for row in rows}
    assert "inp_chlorpyrifos" in restricted
    # Only the Turkish regulation is in force in Konya; the EU instruments apply
    # to the export-facing regions, so the answer must differ by region.
    assert {row["regulation"] for row in rows} == {"reg_tr_ppp"}


async def test_demo_q2_practices_for_wheat_on_clay_that_also_cut_rust(
    writer: Neo4jWriter,
) -> None:
    """Crop -> SoilType <- Practice -> Pest: a branching traversal, not a chain."""
    rows = await writer.read(
        """
        MATCH (c:Crop {id: $crop})-[:SUITED_TO]->(s:SoilType)
        WHERE s.texture CONTAINS 'clay'
        MATCH (pr:Practice)-[:RECOMMENDED_FOR]->(target)
        WHERE target = s OR target = c
        OPTIONAL MATCH (pr)-[m:MITIGATES]->(:Pest {id: $pest})
        RETURN DISTINCT pr.id AS practice, m.mechanism IS NOT NULL AS cuts_rust
        ORDER BY practice
        """,
        {"crop": "crop_wheat", "pest": "pest_yellow_rust"},
    )

    practices = {row["practice"] for row in rows}
    assert {"prac_no_till", "prac_crop_rotation", "prac_resistant_cultivars"} <= practices

    rust_cutting = {row["practice"] for row in rows if row["cuts_rust"]}
    assert rust_cutting == {"prac_resistant_cultivars"}


async def test_climate_zone_hop_is_required_for_zone_questions(writer: Neo4jWriter) -> None:
    """Crop -> Region -> ClimateZone. Modelling climate as a node earns this hop."""
    rows = await writer.read(
        """
        MATCH (c:Crop)-[:GROWN_IN]->(:Region)-[:HAS_CLIMATE]->(z:ClimateZone {id: $zone})
        RETURN DISTINCT c.id AS crop ORDER BY crop
        """,
        {"zone": "clim_humid_subtropical"},
    )
    assert [row["crop"] for row in rows] == ["crop_hazelnut"]
