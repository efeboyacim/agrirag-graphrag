"""Write path into Neo4j.

Deliberately separate from :class:`~agrirag.graph.store.GraphStore`, which is
read-only. Query-time code cannot reach this class, so no generated Cypher can
ever mutate the graph - the capability simply is not wired into the agent.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from neo4j import WRITE_ACCESS, AsyncGraphDatabase

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WriteCounters:
    """What a write actually changed."""

    nodes_created: int = 0
    relationships_created: int = 0
    properties_set: int = 0

    def __add__(self, other: "WriteCounters") -> "WriteCounters":
        return WriteCounters(
            nodes_created=self.nodes_created + other.nodes_created,
            relationships_created=self.relationships_created + other.relationships_created,
            properties_set=self.properties_set + other.properties_set,
        )


class Neo4jWriter:
    """Async write-mode Neo4j client used only by ingestion."""

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j") -> None:
        self._driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        self._database = database

    async def __aenter__(self) -> Self:
        await self._driver.verify_connectivity()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()

    async def execute(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
    ) -> WriteCounters:
        """Run one write statement and return its counters."""
        async with self._driver.session(
            database=self._database,
            default_access_mode=WRITE_ACCESS,
        ) as session:
            result = await session.run(cypher, params or {})
            counters = (await result.consume()).counters
            return WriteCounters(
                nodes_created=counters.nodes_created,
                relationships_created=counters.relationships_created,
                properties_set=counters.properties_set,
            )

    async def apply_migrations(self, path: Path) -> int:
        """Run each statement in a ``.cypher`` file.

        Statements are separated by a line containing only ``;``. Constraints and
        indexes use IF NOT EXISTS, so this is safe to re-run.
        """
        if not path.exists():
            raise FileNotFoundError(f"migration file not found: {path}")

        raw = path.read_text(encoding="utf-8")
        statements = [
            stripped
            for statement in raw.split(";")
            if (
                stripped := "\n".join(
                    line for line in statement.splitlines() if not line.strip().startswith("//")
                ).strip()
            )
        ]

        for statement in statements:
            logger.debug("migration: %s", statement.splitlines()[0])
            await self.execute(statement)
        return len(statements)

    async def count_nodes(self) -> dict[str, int]:
        """Return node counts per label."""
        rows = await self.read(
            "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS c ORDER BY label"
        )
        return {row["label"]: row["c"] for row in rows}

    async def count_relationships(self) -> dict[str, int]:
        """Return relationship counts per type."""
        rows = await self.read(
            "MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS c ORDER BY type"
        )
        return {row["type"]: row["c"] for row in rows}

    async def read(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Read back from the graph. Used by ingestion to verify what it wrote."""
        async with self._driver.session(database=self._database) as session:
            result = await session.run(cypher, params or {})
            return [record.data() async for record in result]

    async def close(self) -> None:
        await self._driver.close()
