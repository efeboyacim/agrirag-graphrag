"""Build the LanceDB index and the graph provenance links.

    uv run agrirag-index                 # embed the corpus into LanceDB
    uv run agrirag-index --extract       # also run LLM extraction into the graph
    uv run agrirag-index --extract-only  # skip embedding, only extract

Embedding is free and offline; extraction costs LLM calls. They are separate
flags so that re-embedding after a corpus edit does not silently spend money.
"""

import argparse
import asyncio
import logging
import sys
import uuid
from pathlib import Path

from agrirag.config import get_settings
from agrirag.ingestion.chunker import chunk_corpus
from agrirag.ingestion.extractor import extract_corpus, write_provenance
from agrirag.ingestion.writer import Neo4jWriter
from agrirag.llm.portkey_client import describe_deployment
from agrirag.observability.logging import configure_logging
from agrirag.vector.embeddings import get_embedder
from agrirag.vector.lancedb_store import LanceDBVectorStore

logger = logging.getLogger("agrirag.index")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CORPUS_DIR = PROJECT_ROOT / "data" / "corpus"


def build_vector_index(corpus_dir: Path = CORPUS_DIR) -> int:
    """Chunk the corpus, embed locally, and rebuild the LanceDB table."""
    documents, chunks = chunk_corpus(corpus_dir)
    if not chunks:
        raise RuntimeError(f"no chunks produced from {corpus_dir}")

    logger.info("chunked %d documents into %d chunks", len(documents), len(chunks))

    embedder = get_embedder()
    vectors = embedder.embed_documents([c.embed_text for c in chunks])
    logger.info("embedded %d chunks with %s", len(vectors), embedder.model_name)

    rows = [
        {
            "vector": vector,
            "chunk_id": chunk.id,
            "doc_id": chunk.doc_id,
            "doc_title": chunk.doc_title,
            "heading": chunk.heading,
            "text": chunk.text,
        }
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]

    store = LanceDBVectorStore()
    count = store.rebuild(rows)
    logger.info("LanceDB table '%s' rebuilt at %s", store.table_name, store.path)
    return count


async def build_provenance(corpus_dir: Path = CORPUS_DIR) -> int:
    """Run LLM extraction and write Document/Chunk/MENTIONS into the graph."""
    settings = get_settings()
    documents, chunks = chunk_corpus(corpus_dir)
    trace_id = f"ingest-{uuid.uuid4().hex[:8]}"

    logger.info("extracting entities from %d chunks via %s", len(chunks), describe_deployment())
    logger.info("trace_id=%s", trace_id)

    mentions = await extract_corpus(chunks, trace_id=trace_id)

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

    logger.info("--- extraction ---")
    logger.info("  chunks processed  : %d", stats.chunks)
    logger.info("  mentions proposed : %d", stats.mentions_proposed)
    logger.info("  mentions linked   : %d (%.0f%%)", stats.mentions_linked, stats.link_rate * 100)
    logger.info("  dropped (unknown) : %d", stats.mentions_unresolved)
    if stats.unresolved_examples:
        logger.info("  examples dropped  : %s", ", ".join(stats.unresolved_examples))

    return stats.mentions_linked


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the AgriRAG retrieval indexes.")
    parser.add_argument(
        "--extract",
        action="store_true",
        help="also run LLM entity extraction and write graph provenance (costs LLM calls)",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="skip embedding; only run extraction",
    )
    args = parser.parse_args()

    configure_logging(get_settings().log_level)

    try:
        if not args.extract_only:
            count = build_vector_index()
            logger.info("vector index: %d chunks", count)

        if args.extract or args.extract_only:
            linked = asyncio.run(build_provenance())
            logger.info("graph provenance: %d MENTIONS edges", linked)
    except Exception as exc:
        logger.error("index build failed: %s", exc, exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
