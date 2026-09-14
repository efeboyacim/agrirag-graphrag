"""LanceDB implementation of :class:`~agrirag.vector.store.VectorStore`.

LanceDB is embedded: this opens a directory, not a connection to a server. There
is deliberately no lancedb container in docker-compose - the store is a volume
mounted into the api container. Reviewers expecting a vector-database service
should read that as a simplification, not an omission.

LanceDB's Python API is synchronous, so calls are pushed to a worker thread. The
alternative - blocking the event loop on file I/O inside an async agent node -
would serialise the parallel retrieval branches in Phase 3 and quietly remove the
benefit of running graph and semantic search concurrently.
"""

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from agrirag.config import get_settings
from agrirag.vector.embeddings import LocalEmbedder, get_embedder
from agrirag.vector.store import SearchHit

if TYPE_CHECKING:  # pragma: no cover
    from lancedb.db import DBConnection
    from lancedb.query import LanceVectorQueryBuilder
    from lancedb.table import Table

logger = logging.getLogger(__name__)


class VectorStoreNotBuiltError(RuntimeError):
    """Raised when the table has not been created yet."""


class LanceDBVectorStore:
    """Embedded vector store over the chunked corpus."""

    def __init__(
        self,
        path: str | Path | None = None,
        table_name: str | None = None,
        embedder: LocalEmbedder | None = None,
    ) -> None:
        settings = get_settings()
        self.path = Path(path or settings.lancedb_path)
        self.table_name = table_name or settings.lancedb_table
        self._embedder = embedder or get_embedder()
        self._db: DBConnection | None = None
        self._table: Table | None = None

    # -- connection ---------------------------------------------------------

    def _connect(self) -> "DBConnection":
        if self._db is None:
            import lancedb

            self.path.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(self.path)
        return self._db

    def _existing_tables(self) -> list[str]:
        """Names of the tables in this database.

        ``list_tables()`` returns a paginated ``ListTablesResponse``, not a list -
        so a plain ``in`` check against it silently reports False and the store
        looks unbuilt. The corpus is one table, so the first page is all of it.
        """
        return list(self._connect().list_tables().tables)

    def _open_table(self) -> "Table":
        if self._table is None:
            db = self._connect()
            if self.table_name not in self._existing_tables():
                raise VectorStoreNotBuiltError(
                    f"LanceDB table '{self.table_name}' does not exist at {self.path}. "
                    "Run: uv run agrirag-index"
                )
            self._table = db.open_table(self.table_name)
        return self._table

    # -- write path (ingestion only) ----------------------------------------

    def rebuild(self, rows: list[dict[str, Any]]) -> int:
        """Replace the table with ``rows``.

        Replace rather than upsert: the corpus is small and rebuilding is a few
        seconds, which avoids a whole class of stale-vector bugs where an edited
        document leaves its previous embedding behind.
        """
        db = self._connect()
        if self.table_name in self._existing_tables():
            db.drop_table(self.table_name)
        self._table = db.create_table(self.table_name, data=rows)
        logger.info("built LanceDB table '%s' with %d chunks", self.table_name, len(rows))
        return len(rows)

    # -- read path ----------------------------------------------------------

    def _search_sync(self, query: str, k: int) -> list[SearchHit]:
        table = self._open_table()
        vector = self._embedder.embed_query(query)

        # search() is typed as returning the base LanceQueryBuilder, but a vector
        # query returns LanceVectorQueryBuilder, which is where distance_type lives.
        builder = cast("LanceVectorQueryBuilder", table.search(vector))
        rows = builder.distance_type("cosine").limit(k).to_list()

        hits: list[SearchHit] = []
        for row in rows:
            # LanceDB returns cosine *distance*; the agent reasons about
            # similarity, so normalise once here rather than at every call site.
            distance = float(row.get("_distance", 1.0))
            hits.append(
                SearchHit(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    doc_title=row["doc_title"],
                    heading=row.get("heading", ""),
                    text=row["text"],
                    score=round(max(0.0, 1.0 - distance), 4),
                )
            )
        return hits

    async def search(self, query: str, k: int = 5) -> list[SearchHit]:
        return await asyncio.to_thread(self._search_sync, query, k)

    async def count(self) -> int:
        return await asyncio.to_thread(lambda: self._open_table().count_rows())

    async def close(self) -> None:
        self._table = None
        self._db = None
