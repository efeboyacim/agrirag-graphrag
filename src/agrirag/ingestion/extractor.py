"""LLM entity extraction over the document corpus.

This is the *additive* half of the hybrid graph-construction strategy. The
deterministic CSV backbone in :mod:`agrirag.ingestion.graph_builder` defines the
agronomic facts and never changes; this pass reads the prose corpus and links
each chunk to the entities it discusses:

    (:Chunk)-[:MENTIONS {confidence}]->(:Crop|:Pest|:Input|:Practice|...)
    (:Chunk)-[:PART_OF]->(:Document)

Why additive rather than authoritative: an extraction pass that could *create*
entities would make the graph non-deterministic between runs, and the Phase 5
eval thresholds would drift with every re-ingestion. So extraction may only link
to entities that already exist. A mention the model invents is dropped, and that
is recorded rather than silently ignored.

What this buys: provenance. A semantic hit can be traced to the graph entities it
concerns, and a graph entity can be traced back to the prose explaining it - which
is what makes the "both" retrieval route worth having.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from agrirag.graph.schema import FULLTEXT_LABELS
from agrirag.ingestion.chunker import Chunk
from agrirag.ingestion.writer import Neo4jWriter
from agrirag.llm.portkey_client import Span, structured_model

logger = logging.getLogger(__name__)

#: How many chunks to extract concurrently. Enough to be quick, low enough that a
#: rate limit is unlikely - and the gateway retries if one occurs anyway.
CONCURRENCY = 5


#: Constrained as a Literal rather than a free string. A description alone is a
#: suggestion the model ignores - a probe returned labels like "PATHOGEN_OR_AGENT"
#: and "PEST_STAGE". As a Literal it becomes part of the tool schema, so the
#: provider enforces it and every label is guaranteed to exist in the graph.
EntityLabel = Literal["Crop", "Region", "SoilType", "Pest", "Input", "Practice"]


class ExtractedMention(BaseModel):
    """One entity the model believes a chunk discusses."""

    name: str = Field(description="Entity name exactly as written in the text")
    label: EntityLabel = Field(description="Which kind of entity this is")
    confidence: float = Field(ge=0.0, le=1.0, description="0-1 confidence")


class ExtractionResult(BaseModel):
    """Structured output for one chunk."""

    mentions: list[ExtractedMention] = Field(default_factory=list)

    @field_validator("mentions", mode="before")
    @classmethod
    def _accept_stringified_list(cls, value: object) -> object:
        """Tolerate a nested array arriving as a JSON string.

        Going through the gateway, Anthropic intermittently returns the nested
        tool argument as ``'{"mentions": [...]}'`` or ``'[...]'`` - a string
        rather than an array. It is not consistent: the same prompt succeeds on
        one call and stringifies on the next, which makes it a poor thing to
        discover in a 107-chunk batch run.

        Parsing here rather than failing the chunk keeps one translation quirk
        from silently emptying the provenance graph.
        """
        if not isinstance(value, str):
            return value
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            logger.warning("could not parse stringified mentions: %.80s", value)
            return []
        if isinstance(parsed, dict):
            return parsed.get("mentions", [])
        return parsed


SYSTEM_PROMPT = """You extract agricultural entity mentions from text.

Return every entity the passage genuinely discusses, using these labels:
{labels}

Rules:
- Only list entities the passage actually talks about, not ones merely implied.
- Use the name as written in the passage.
- A passage explaining a concept in general terms may legitimately mention nothing.
- Do not invent entities to fill the list.
- confidence: 1.0 when the passage is substantially about the entity, lower when
  it is mentioned in passing."""


@dataclass
class ExtractionStats:
    """What an extraction run did - reported so a silent failure is visible."""

    chunks: int = 0
    mentions_proposed: int = 0
    mentions_linked: int = 0
    mentions_unresolved: int = 0
    unresolved_examples: list[str] | None = None

    @property
    def link_rate(self) -> float:
        if not self.mentions_proposed:
            return 0.0
        return round(self.mentions_linked / self.mentions_proposed, 3)


async def extract_chunk(chunk: Chunk, *, trace_id: str) -> list[ExtractedMention]:
    """Extract entity mentions from one chunk via the gateway."""
    llm = structured_model(
        ExtractionResult,
        span=Span.EXTRACT,
        trace_id=trace_id,
        max_tokens=800,
        metadata={"chunk_id": chunk.id, "doc_id": chunk.doc_id},
    )

    try:
        result = await llm.ainvoke(
            [
                ("system", SYSTEM_PROMPT.format(labels=", ".join(FULLTEXT_LABELS))),
                ("human", f"Passage:\n\n{chunk.embed_text}"),
            ]
        )
    except Exception as exc:
        logger.warning("extraction failed for %s: %s", chunk.id, exc)
        return []

    if isinstance(result, ExtractionResult):
        return result.mentions
    try:
        return ExtractionResult.model_validate(result).mentions
    except ValidationError as exc:
        logger.warning("unparseable extraction for %s: %s", chunk.id, exc)
        return []


async def extract_corpus(
    chunks: list[Chunk], *, trace_id: str
) -> dict[str, list[ExtractedMention]]:
    """Extract mentions for every chunk, bounded concurrency."""
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def one(chunk: Chunk) -> tuple[str, list[ExtractedMention]]:
        async with semaphore:
            return chunk.id, await extract_chunk(chunk, trace_id=trace_id)

    results = await asyncio.gather(*(one(chunk) for chunk in chunks))
    return dict(results)


async def write_provenance(
    writer: Neo4jWriter,
    documents: list[tuple[str, str, str]],
    chunks: list[Chunk],
    mentions: dict[str, list[ExtractedMention]],
) -> ExtractionStats:
    """Write Document, Chunk, PART_OF and MENTIONS into the graph.

    MENTIONS is resolved against existing nodes by exact name or alias match.
    An unresolved mention is counted and dropped - never created - so the
    backbone stays exactly as the CSVs defined it.
    """
    await writer.execute(
        "UNWIND $rows AS row MERGE (d:Document {id: row.id}) SET d += row",
        {"rows": [{"id": d[0], "title": d[1], "source": d[2]} for d in documents]},
    )

    await writer.execute(
        """
        UNWIND $rows AS row
        MERGE (c:Chunk {id: row.id})
        SET c.text = row.text, c.ordinal = row.ordinal, c.heading = row.heading
        WITH c, row
        MATCH (d:Document {id: row.doc_id})
        MERGE (c)-[:PART_OF]->(d)
        """,
        {
            "rows": [
                {
                    "id": c.id,
                    "doc_id": c.doc_id,
                    "text": c.text,
                    "ordinal": c.ordinal,
                    "heading": c.heading,
                }
                for c in chunks
            ]
        },
    )

    rows: list[dict[str, Any]] = [
        {"chunk_id": chunk_id, "name": m.name.strip(), "confidence": m.confidence}
        for chunk_id, chunk_mentions in mentions.items()
        for m in chunk_mentions
    ]

    stats = ExtractionStats(chunks=len(chunks), mentions_proposed=len(rows))
    if not rows:
        return stats

    # Resolve against every name-like property, case-insensitively:
    #   name                -> "Fall armyworm"
    #   aliases             -> "sune", "misir"           (incl. Turkish names)
    #   scientific_name     -> "Bactrocera oleae"        (the model prefers these)
    #   active_ingredient   -> "Bacillus thuringiensis subsp. kurstaki"
    #
    # Matching only name+aliases dropped a large share of correct mentions,
    # because prose about a pest routinely uses the binomial and prose about a
    # product uses the active ingredient rather than the trade name.
    #
    # Matching stays exact (after lowercasing). Fuzzy matching would recover a
    # few more - "Black Sea" for "Black Sea Coast" - at the cost of false links
    # that are invisible until they corrupt a retrieval result. An unlinked
    # mention is a counted, visible miss; a wrongly linked one is not.
    linked = await writer.read(
        """
        UNWIND $rows AS row
        MATCH (c:Chunk {id: row.chunk_id})
        MATCH (e)
        WHERE any(l IN labels(e) WHERE l IN $labels)
          AND toLower(row.name) IN [
                toLower(coalesce(e.name, '')),
                toLower(coalesce(e.scientific_name, '')),
                toLower(coalesce(e.active_ingredient, ''))
              ] + [a IN coalesce(e.aliases, []) | toLower(a)]
        MERGE (c)-[m:MENTIONS]->(e)
        SET m.confidence = row.confidence
        RETURN row.name AS name, e.id AS entity_id
        """,
        {"rows": rows, "labels": list(FULLTEXT_LABELS)},
    )

    resolved_names = {str(r["name"]).lower() for r in linked}
    unresolved: list[str] = sorted(
        {str(r["name"]) for r in rows if str(r["name"]).lower() not in resolved_names}
    )

    stats.mentions_linked = len(linked)
    stats.mentions_unresolved = len(unresolved)
    stats.unresolved_examples = unresolved[:10]

    if unresolved:
        logger.info(
            "%d proposed mentions did not match a seeded entity and were dropped: %s",
            len(unresolved),
            json.dumps(unresolved[:10]),
        )

    return stats
