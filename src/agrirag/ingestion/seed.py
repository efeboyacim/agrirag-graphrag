"""Seed the knowledge graph from the committed CSVs.

    uv run agrirag-seed            # load into the running Neo4j container
    uv run agrirag-seed --reset    # wipe the graph first
    uv run agrirag-seed --dry-run  # parse and validate the CSVs, touch nothing

Safe to re-run: every write is a MERGE keyed on ``id``.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from agrirag.config import get_settings
from agrirag.graph.schema import NODE_SPECS, REL_SPECS
from agrirag.ingestion.graph_builder import (
    build_migrations,
    load_nodes,
    load_relationships,
    verify_edges,
)
from agrirag.ingestion.loader import SeedDataError, read_rows
from agrirag.ingestion.writer import Neo4jWriter
from agrirag.observability.logging import configure_logging

logger = logging.getLogger("agrirag.seed")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SEED_DIR = PROJECT_ROOT / "data" / "seed"
MIGRATIONS_PATH = PROJECT_ROOT / "src" / "agrirag" / "graph" / "migrations.cypher"


def validate(seed_dir: Path) -> tuple[int, int]:
    """Parse every seed file without touching the database.

    Returns ``(node_rows, edge_rows)``. Raises :class:`SeedDataError` on the first
    malformed file, which is what makes ``--dry-run`` useful in CI.
    """
    node_rows = 0
    for spec in NODE_SPECS:
        rows = read_rows(seed_dir / spec.source, spec.casts)
        ids = [row["id"] for row in rows]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise SeedDataError(f"{spec.source}: duplicate ids {sorted(duplicates)}")
        node_rows += len(rows)
        logger.info("%-12s %3d rows", spec.label, len(rows))

    edge_rows = 0
    for rel in REL_SPECS:
        rows = read_rows(seed_dir / rel.source, rel.casts)
        edge_rows += len(rows)
        logger.info("%-16s %3d rows", rel.type, len(rows))

    return node_rows, edge_rows


async def seed(*, reset: bool, seed_dir: Path = SEED_DIR) -> int:
    """Apply migrations and load the graph. Returns a process exit code."""
    settings = get_settings()

    async with Neo4jWriter(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
        database=settings.neo4j_database,
    ) as writer:
        if reset:
            logger.warning("--reset: deleting every node and relationship")
            await writer.execute("MATCH (n) DETACH DELETE n")

        MIGRATIONS_PATH.write_text(build_migrations(), encoding="utf-8")
        applied = await writer.apply_migrations(MIGRATIONS_PATH)
        logger.info("applied %d migration statements", applied)

        for node_spec in NODE_SPECS:
            await load_nodes(writer, node_spec, seed_dir)

        for rel_spec in REL_SPECS:
            await load_relationships(writer, rel_spec, seed_dir)

        problems = await verify_edges(writer, seed_dir)
        if problems:
            for problem in problems:
                logger.error("edge verification failed: %s", problem)
            return 1

        nodes = await writer.count_nodes()
        rels = await writer.count_relationships()

        logger.info("--- graph loaded ---")
        for label, count in sorted(nodes.items()):
            logger.info("  (:%s) %d", label, count)
        for rel_type, count in sorted(rels.items()):
            logger.info("  [:%s] %d", rel_type, count)
        logger.info("total: %d nodes, %d relationships", sum(nodes.values()), sum(rels.values()))

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed the AgriRAG knowledge graph.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="delete all existing nodes and relationships before loading",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate the seed CSVs without connecting to Neo4j",
    )
    args = parser.parse_args()

    configure_logging(get_settings().log_level)

    try:
        if args.dry_run:
            nodes, edges = validate(SEED_DIR)
            logger.info("dry run OK: %d node rows, %d edge rows", nodes, edges)
            return 0
        return asyncio.run(seed(reset=args.reset))
    except SeedDataError as exc:
        logger.error("seed data invalid: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
