"""Index rebuilding.

Runs synchronously and returns real counts. A background-job API with a status
endpoint would be more correct for a long operation, but this is an
administrative endpoint on a demo system, and a half-built job queue is harder
to explain than a slow request. The duration is in the response so nobody is
surprised twice.
"""

import asyncio
import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends

from agrirag.api.deps import get_request_id, get_vector_store
from agrirag.api.schemas import ErrorResponse, IngestRequest, IngestResponse
from agrirag.config import get_settings
from agrirag.ingestion.chunker import chunk_corpus
from agrirag.ingestion.extractor import extract_corpus, write_provenance
from agrirag.ingestion.index_builder import CORPUS_DIR, build_vector_index
from agrirag.ingestion.writer import Neo4jWriter
from agrirag.vector.store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ingest"])


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Rebuild the retrieval indexes from the corpus",
    responses={
        503: {"model": ErrorResponse, "description": "A backing store is unavailable."},
    },
)
async def ingest(
    payload: IngestRequest,
    vectors: Annotated[VectorStore, Depends(get_vector_store)],
    request_id: Annotated[str, Depends(get_request_id)],
) -> IngestResponse:
    """Re-chunk and re-embed the corpus, and optionally re-extract entities.

    `rebuild_vectors` is free and offline: embeddings run locally, so this costs
    nothing but about thirty seconds.

    `extract_entities` costs one LLM call per chunk (~107 for this corpus) and
    writes `(:Chunk)-[:MENTIONS]->(:Entity)` provenance. Extraction only ever
    *links* to entities that already exist - it cannot create them - so the
    deterministic CSV backbone stays byte-identical and the evaluation baseline
    does not drift when the corpus is re-ingested. Mentions that match nothing
    are counted and dropped, and the count is returned rather than swallowed.
    """
    started = time.perf_counter()
    settings = get_settings()
    documents, chunks = chunk_corpus(CORPUS_DIR)

    response = IngestResponse(
        documents=len(documents),
        chunks=len(chunks),
        request_id=request_id,
    )

    if payload.rebuild_vectors:
        # Embedding is CPU-bound and synchronous; off the event loop it goes.
        await asyncio.to_thread(build_vector_index, CORPUS_DIR)
        # The app's store caches an open table handle that now points at a
        # dropped table. Closing resets it so the next search reopens.
        await vectors.close()
        response.vectors_rebuilt = True
        logger.info("vector index rebuilt (%d chunks)", len(chunks))

    if payload.extract_entities:
        mentions = await extract_corpus(chunks, trace_id=request_id)

        # Ingestion uses its own writer. The agent's GraphStore is read-only by
        # design, and this endpoint does not get to break that.
        async with Neo4jWriter(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        ) as writer:
            stats = await write_provenance(
                writer,
                [(d.id, d.title, d.source) for d in documents],
                chunks,
                mentions,
            )

        response.extraction_ran = True
        response.mentions_proposed = stats.mentions_proposed
        response.mentions_linked = stats.mentions_linked
        response.mentions_dropped = stats.mentions_unresolved
        response.dropped_examples = stats.unresolved_examples or []
        logger.info(
            "extraction linked %d/%d mentions (%.0f%%)",
            stats.mentions_linked,
            stats.mentions_proposed,
            stats.link_rate * 100,
        )

    response.duration_seconds = round(time.perf_counter() - started, 2)
    return response
