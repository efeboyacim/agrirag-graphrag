"""Neo4j implementation of :class:`~agrirag.graph.store.GraphStore`.

Phase 0 provides connectivity plus a generic read. Phase 1 adds schema
migrations, Phase 2 adds the parameterised query templates and the Cypher
guard that sits in front of ``read``.
"""

import logging
from typing import Any

from neo4j import READ_ACCESS, AsyncGraphDatabase

logger = logging.getLogger(__name__)


class Neo4jGraphStore:
    """Async Neo4j-backed graph store."""

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
    ) -> None:
        self._driver = AsyncGraphDatabase.driver(uri, auth=(user, password))
        self._database = database
        logger.debug("Neo4j driver created for %s (database=%s)", uri, database)

    async def verify_connectivity(self) -> None:
        await self._driver.verify_connectivity()

    async def read(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        # READ_ACCESS is a second line of defence behind the Cypher guard:
        # even a write statement that slipped through is rejected by the server.
        async with self._driver.session(
            database=self._database,
            default_access_mode=READ_ACCESS,
        ) as session:
            result = await session.run(cypher, params or {})
            return [record.data() async for record in result]

    async def close(self) -> None:
        await self._driver.close()
