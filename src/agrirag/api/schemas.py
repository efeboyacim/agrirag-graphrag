"""Pydantic v2 request/response models shared across API versions."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from agrirag.agent.state import AgentResult, Route

DependencyStatus = Literal["ok", "unavailable"]


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Liveness plus per-dependency readiness.

    Each dependency is reported separately rather than collapsed into one flag,
    because they fail for different reasons and have different fixes: Neo4j means
    the container is down, LanceDB means the index was never built
    (``uv run agrirag-index``), and the gateway means LLM calls will fail while
    retrieval still works.
    """

    status: Literal["ok", "degraded"] = Field(
        description="'ok' when every dependency is reachable, otherwise 'degraded'."
    )
    version: str = Field(description="Running application version.")
    neo4j: DependencyStatus = Field(description="Graph store reachability.")
    lancedb: DependencyStatus = Field(description="Vector index built and readable.")
    gateway: DependencyStatus = Field(description="Portkey LLM gateway reachability.")
    chunks: int = Field(default=0, description="Chunks in the vector index.")
    llm_routing: str = Field(description="How LLM traffic is currently routed.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "status": "ok",
                    "version": "0.1.0",
                    "neo4j": "ok",
                    "lancedb": "ok",
                    "gateway": "ok",
                    "chunks": 107,
                    "llm_routing": "Portkey self-hosted OSS gateway at http://gateway:8787/v1",
                }
            ]
        }
    }


# --------------------------------------------------------------------------
# Ask
# --------------------------------------------------------------------------


class AskRequest(BaseModel):
    """A question for the agent."""

    question: str = Field(
        min_length=3,
        max_length=1000,
        description="A natural-language agricultural question, in English or Turkish.",
    )
    thread_id: str | None = Field(
        default=None,
        description=(
            "Conversation thread. Reusing a thread_id persists agent state across "
            "turns via the SQLite checkpointer."
        ),
    )
    force_route: Route | None = Field(
        default=None,
        description=(
            "Skip the routing node and use this path instead. Intended for "
            "comparison: forcing 'semantic' on a multi-hop question shows what "
            "the graph contributes."
        ),
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": (
                        "Which organic-approved treatments exist for maize pests in Konya, "
                        "and are any of them restricted there?"
                    )
                },
                {
                    "question": "Explain the principles behind integrated pest management.",
                },
                {
                    "question": "Which organic-approved treatments exist for maize pests in Konya?",
                    "force_route": "semantic",
                },
            ]
        }
    }


class CitationOut(BaseModel):
    """Where a piece of the answer came from."""

    kind: Literal["graph", "document"]
    ref: str = Field(description="Node id for graph facts, chunk id for documents.")
    label: str


class AskResponse(BaseModel):
    """The agent's answer, plus everything needed to audit how it got there.

    ``retrieval_context`` is returned deliberately. It is what the Phase 5
    retrieval metrics score, and exposing it means a reviewer can see exactly
    what the answer was grounded in rather than taking it on trust.
    """

    question: str
    answer: str
    abstained: bool = Field(description="True when the agent declined to answer.")

    route: Route
    route_reason: str
    retrieval_context: list[str] = Field(default_factory=list)
    citations: list[CitationOut] = Field(default_factory=list)

    # -- traceability -----------------------------------------------------
    request_id: str = Field(description="Correlates with the Portkey trace for this request.")
    template: str | None = Field(default=None, description="Cypher template used, if any.")
    cypher: str | None = Field(default=None, description="The query actually executed.")
    linked_entities: list[dict[str, Any]] = Field(default_factory=list)
    graph_row_count: int = 0
    chunk_count: int = 0
    retries: int = 0

    @classmethod
    def from_result(cls, result: AgentResult, request_id: str) -> "AskResponse":
        return cls(
            question=result.question,
            answer=result.answer,
            abstained=result.abstained,
            route=result.route,
            route_reason=result.route_reason,
            retrieval_context=result.retrieval_context,
            citations=[CitationOut(**c.model_dump()) for c in result.citations],
            request_id=request_id,
            template=result.template,
            cypher=result.cypher,
            linked_entities=result.linked_entities,
            graph_row_count=result.graph_row_count,
            chunk_count=result.chunk_count,
            retries=result.retries,
        )


# --------------------------------------------------------------------------
# Graph schema
# --------------------------------------------------------------------------


class RelationshipOut(BaseModel):
    """One relationship type in the schema."""

    type: str
    start: str
    end: str
    properties: list[str] = Field(default_factory=list)
    description: str
    count: int = 0


class NodeLabelOut(BaseModel):
    """One node label in the schema."""

    label: str
    description: str
    properties: list[str] = Field(default_factory=list)
    count: int = 0


class GraphSchemaResponse(BaseModel):
    """The live graph schema with current counts."""

    labels: list[NodeLabelOut]
    relationships: list[RelationshipOut]
    total_nodes: int
    total_relationships: int
    prompt_schema: str = Field(
        description="The compact schema string injected into the agent's prompts."
    )


# --------------------------------------------------------------------------
# Ingest
# --------------------------------------------------------------------------


class IngestRequest(BaseModel):
    """Rebuild the retrieval indexes from the corpus on disk."""

    rebuild_vectors: bool = Field(
        default=True, description="Re-chunk and re-embed the corpus into LanceDB. Free, offline."
    )
    extract_entities: bool = Field(
        default=False,
        description=(
            "Also run LLM entity extraction and write (:Chunk)-[:MENTIONS]->(:Entity) "
            "provenance. Costs one LLM call per chunk."
        ),
    )

    model_config = {
        "json_schema_extra": {"examples": [{"rebuild_vectors": True, "extract_entities": False}]}
    }


class IngestResponse(BaseModel):
    """What an ingestion run did."""

    documents: int = 0
    chunks: int = 0
    vectors_rebuilt: bool = False
    extraction_ran: bool = False
    mentions_proposed: int = 0
    mentions_linked: int = 0
    mentions_dropped: int = 0
    dropped_examples: list[str] = Field(
        default_factory=list,
        description="Names the model proposed that match no seeded entity. Dropped, never created.",
    )
    duration_seconds: float = 0.0
    request_id: str = ""


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """A problem, reported without leaking internals."""

    detail: str = Field(description="What went wrong, in terms the caller can act on.")
    error_type: str = Field(description="Stable machine-readable category.")
    request_id: str = Field(description="Quote this when reporting the problem.")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "detail": (
                        "The LLM gateway is not reachable. Check that the gateway "
                        "container is running and ANTHROPIC_API_KEY is set."
                    ),
                    "error_type": "llm_unavailable",
                    "request_id": "c0ffee00-1234-5678-9abc-def012345678",
                }
            ]
        }
    }
