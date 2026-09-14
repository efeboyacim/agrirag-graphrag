"""Seed loading, type coercion and schema consistency.

These run without Neo4j, so CI catches a malformed CSV or a schema/data mismatch
before anything tries to connect to a database.
"""

from pathlib import Path

import pytest

from agrirag.graph.schema import (
    ALL_LABELS,
    ALL_REL_TYPES,
    NODE_SPECS,
    REL_SPECS,
    schema_summary,
)
from agrirag.ingestion.graph_builder import _reshape, build_migrations
from agrirag.ingestion.loader import SeedDataError, read_rows
from agrirag.ingestion.seed import SEED_DIR, validate

# --------------------------------------------------------------------------
# Loader
# --------------------------------------------------------------------------


def test_casts_are_applied(tmp_path: Path) -> None:
    path = tmp_path / "t.csv"
    path.write_text("id,n,f,b,l\nx1,42,3.5,true,a|b|c\n", encoding="utf-8")

    row = read_rows(path, {"n": "int", "f": "float", "b": "bool", "l": "list"})[0]

    assert row == {"id": "x1", "n": 42, "f": 3.5, "b": True, "l": ["a", "b", "c"]}


def test_empty_cells_are_dropped_rather_than_stored_as_empty_strings(tmp_path: Path) -> None:
    """A property that does not apply should be absent, not an empty string."""
    path = tmp_path / "t.csv"
    path.write_text("id,npk\ninp_x,\n", encoding="utf-8")

    assert read_rows(path) == [{"id": "inp_x"}]


@pytest.mark.parametrize("bad", ["notanumber", "1.2.3"])
def test_a_bad_numeric_cast_names_the_file_and_field(tmp_path: Path, bad: str) -> None:
    path = tmp_path / "broken.csv"
    path.write_text(f"id,n\nx1,{bad}\n", encoding="utf-8")

    with pytest.raises(SeedDataError) as exc:
        read_rows(path, {"n": "int"})

    assert "broken.csv" in str(exc.value)
    assert "n=" in str(exc.value)


def test_a_missing_seed_file_is_reported_clearly(tmp_path: Path) -> None:
    with pytest.raises(SeedDataError, match="seed file not found"):
        read_rows(tmp_path / "absent.csv")


# --------------------------------------------------------------------------
# Schema / data consistency
# --------------------------------------------------------------------------


def test_every_spec_points_at_a_file_that_exists() -> None:
    for spec in NODE_SPECS:
        assert (SEED_DIR / spec.source).exists(), spec.source
    for spec in REL_SPECS:
        assert (SEED_DIR / spec.source).exists(), spec.source


def test_relationship_endpoints_reference_declared_labels() -> None:
    for spec in REL_SPECS:
        assert spec.start_label in ALL_LABELS, spec.type
        ends = spec.end_label_options if spec.end_label_field else (spec.end_label,)
        for end in ends:
            assert end in ALL_LABELS, f"{spec.type} -> {end}"


def test_every_edge_id_resolves_to_a_seeded_node() -> None:
    """Edges load with MATCH, so an unknown id vanishes silently. Catch it here."""
    known: dict[str, set[str]] = {
        spec.label: {row["id"] for row in read_rows(SEED_DIR / spec.source, spec.casts)}
        for spec in NODE_SPECS
    }

    for spec in REL_SPECS:
        rows = read_rows(SEED_DIR / spec.source, spec.casts)
        for end_label, group in _reshape(spec, rows).items():
            for edge in group:
                assert edge["start"] in known[spec.start_label], (
                    f"{spec.source}: unknown {spec.start_label} id {edge['start']}"
                )
                assert edge["end"] in known[end_label], (
                    f"{spec.source}: unknown {end_label} id {edge['end']}"
                )


def test_declared_columns_are_present_in_the_csv_headers() -> None:
    for spec in REL_SPECS:
        row = read_rows(SEED_DIR / spec.source, spec.casts)[0]
        assert spec.start_field in row, f"{spec.source} missing {spec.start_field}"
        assert spec.end_field in row, f"{spec.source} missing {spec.end_field}"


def test_validate_finds_no_duplicate_ids() -> None:
    node_rows, edge_rows = validate(SEED_DIR)

    assert node_rows == 75
    assert edge_rows == 190


def test_organic_flag_parses_as_a_real_boolean() -> None:
    """Cypher filters on `i.organic_approved = true`, so the string 'false' would
    quietly match as truthy and break the headline demo query."""
    rows = {r["id"]: r for r in read_rows(SEED_DIR / "inputs.csv", {"organic_approved": "bool"})}

    assert rows["inp_bt"]["organic_approved"] is True
    assert rows["inp_chlorpyrifos"]["organic_approved"] is False


# --------------------------------------------------------------------------
# Generated artefacts
# --------------------------------------------------------------------------


def test_migrations_cover_every_label_and_the_fulltext_index() -> None:
    migrations = build_migrations()

    for label in ALL_LABELS:
        assert f"FOR (n:{label}) REQUIRE n.id IS UNIQUE" in migrations
    assert "CREATE FULLTEXT INDEX entity_names" in migrations
    # Provenance labels are constrained ahead of Phase 2 populating them.
    assert "FOR (n:Chunk) REQUIRE n.id IS UNIQUE" in migrations


def test_schema_summary_mentions_every_label_and_relationship() -> None:
    """This string is what the Phase 2 router and Cypher prompts actually see."""
    summary = schema_summary()

    for label in ALL_LABELS:
        assert f"(:{label}" in summary
    for rel in ALL_REL_TYPES:
        assert f"[:{rel}" in summary


def test_polymorphic_relationship_groups_by_target_label() -> None:
    spec = next(s for s in REL_SPECS if s.type == "RECOMMENDED_FOR")
    rows = read_rows(SEED_DIR / spec.source, spec.casts)

    grouped = _reshape(spec, rows)

    assert set(grouped) == {"Crop", "SoilType"}
    assert sum(len(g) for g in grouped.values()) == len(rows)
