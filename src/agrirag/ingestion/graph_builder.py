"""Deterministic graph construction from the seed CSVs.

This is the backbone loader: every node and edge here comes from a committed CSV,
so the graph is byte-identical on every run. That stability is what lets the
Phase 5 eval thresholds mean something.

Phase 2 layers LLM-extracted ``(:Chunk)-[:MENTIONS]->(:Entity)`` provenance on
top of this backbone without altering it.
"""

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from agrirag.graph.schema import (
    FULLTEXT_INDEX_NAME,
    FULLTEXT_LABELS,
    FULLTEXT_PROPERTIES,
    NODE_SPECS,
    REL_SPECS,
    NodeSpec,
    RelSpec,
)
from agrirag.ingestion.loader import SeedDataError, iter_batches, read_rows
from agrirag.ingestion.writer import Neo4jWriter, WriteCounters

logger = logging.getLogger(__name__)

BATCH_SIZE = 500


def build_migrations() -> str:
    """Generate the constraint and index statements from the schema.

    Derived rather than hand-written, so a new label cannot be added to
    :mod:`agrirag.graph.schema` while silently missing its uniqueness constraint.
    """
    lines = ["// Generated from agrirag.graph.schema - do not edit by hand.", ""]

    for spec in NODE_SPECS:
        lines.append(f"// {spec.label}")
        lines.append(
            f"CREATE CONSTRAINT {spec.label.lower()}_id_unique IF NOT EXISTS "
            f"FOR (n:{spec.label}) REQUIRE n.id IS UNIQUE"
        )
        lines.append(";")

    lines.append("")
    lines.append("// Provenance nodes (populated in Phase 2)")
    for label in ("Document", "Chunk"):
        lines.append(
            f"CREATE CONSTRAINT {label.lower()}_id_unique IF NOT EXISTS "
            f"FOR (n:{label}) REQUIRE n.id IS UNIQUE"
        )
        lines.append(";")

    lines.append("")
    lines.append("// Fulltext index backing entity linking.")
    lines.append("// Dropped first: CREATE ... IF NOT EXISTS silently keeps the old")
    lines.append("// definition, so adding a label to FULLTEXT_LABELS would have no")
    lines.append("// effect and the new label would stay permanently unlinkable.")
    lines.append(f"DROP INDEX {FULLTEXT_INDEX_NAME} IF EXISTS")
    lines.append(";")
    labels = "|".join(FULLTEXT_LABELS)
    props = ", ".join(f"n.{p}" for p in FULLTEXT_PROPERTIES)
    lines.append(
        f"CREATE FULLTEXT INDEX {FULLTEXT_INDEX_NAME} IF NOT EXISTS "
        f"FOR (n:{labels}) ON EACH [{props}]"
    )
    lines.append(";")

    return "\n".join(lines) + "\n"


async def load_nodes(writer: Neo4jWriter, spec: NodeSpec, seed_dir: Path) -> WriteCounters:
    """MERGE every row of a node CSV into the graph."""
    rows = read_rows(seed_dir / spec.source, spec.casts)
    if not rows:
        raise SeedDataError(f"{spec.source} produced no rows")

    cypher = f"UNWIND $rows AS row MERGE (n:{spec.label} {{id: row.id}}) SET n += row"

    total = WriteCounters()
    for batch in iter_batches(rows, BATCH_SIZE):
        total += await writer.execute(cypher, {"rows": batch})

    logger.info("%-12s %3d rows -> %d nodes created", spec.label, len(rows), total.nodes_created)
    return total


def _reshape(spec: RelSpec, rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Split relationship rows into ``{end_label: [{start, end, props}, ...]}``.

    Cypher cannot parameterise a label, so a relationship whose target label
    varies per row (RECOMMENDED_FOR) is grouped and run once per label.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        if spec.end_label_field:
            end_label = row.get(spec.end_label_field)
            if end_label not in spec.end_label_options:
                raise SeedDataError(
                    f"{spec.source}: {spec.end_label_field}={end_label!r} is not one of "
                    f"{spec.end_label_options}"
                )
        else:
            end_label = spec.end_label

        grouped[str(end_label)].append(
            {
                "start": row[spec.start_field],
                "end": row[spec.end_field],
                "props": {p: row[p] for p in spec.properties if p in row},
            }
        )

    return dict(grouped)


async def load_relationships(writer: Neo4jWriter, spec: RelSpec, seed_dir: Path) -> WriteCounters:
    """MERGE every row of an edge CSV into the graph.

    Uses MATCH (not MERGE) on the endpoints: an edge referencing an unknown id is
    a data error and should surface as a missing relationship, not conjure an
    empty node into existence.
    """
    rows = read_rows(seed_dir / spec.source, spec.casts)
    if not rows:
        raise SeedDataError(f"{spec.source} produced no rows")

    total = WriteCounters()
    for end_label, group in _reshape(spec, rows).items():
        cypher = (
            "UNWIND $rows AS row "
            f"MATCH (a:{spec.start_label} {{id: row.start}}) "
            f"MATCH (b:{end_label} {{id: row.end}}) "
            f"MERGE (a)-[r:{spec.type}]->(b) "
            "SET r += row.props"
        )
        for batch in iter_batches(group, BATCH_SIZE):
            total += await writer.execute(cypher, {"rows": batch})

    expected = len(rows)
    if total.relationships_created and total.relationships_created < expected:
        logger.warning(
            "%s: %d of %d edges created - the rest already existed or reference unknown ids",
            spec.type,
            total.relationships_created,
            expected,
        )
    logger.info(
        "%-16s %3d rows -> %d rels created",
        spec.type,
        expected,
        total.relationships_created,
    )
    return total


async def verify_edges(writer: Neo4jWriter, seed_dir: Path) -> list[str]:
    """Return a list of edge rows whose endpoints are missing from the graph.

    Catches the silent failure mode of MATCH-based edge loading: a typo in an id
    produces no error, just a quietly absent relationship.
    """
    problems: list[str] = []

    for spec in REL_SPECS:
        rows = read_rows(seed_dir / spec.source, spec.casts)
        for end_label, group in _reshape(spec, rows).items():
            found = await writer.read(
                "UNWIND $rows AS row "
                f"MATCH (a:{spec.start_label} {{id: row.start}}) "
                f"MATCH (b:{end_label} {{id: row.end}}) "
                f"MATCH (a)-[r:{spec.type}]->(b) "
                "RETURN count(r) AS c",
                {"rows": group},
            )
            actual = found[0]["c"] if found else 0
            if actual != len(group):
                problems.append(
                    f"{spec.type} -> {end_label}: expected {len(group)} edges, graph has {actual}"
                )

    return problems
