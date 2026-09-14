"""Graph schema introspection."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends

from agrirag.api.deps import get_graph_store
from agrirag.api.schemas import GraphSchemaResponse, NodeLabelOut, RelationshipOut
from agrirag.graph.schema import NODE_SPECS, REL_SPECS, schema_summary
from agrirag.graph.store import GraphStore

logger = logging.getLogger(__name__)

router = APIRouter(tags=["graph"])


@router.get(
    "/graph/schema",
    response_model=GraphSchemaResponse,
    summary="The knowledge graph schema with live counts",
)
async def graph_schema(
    store: Annotated[GraphStore, Depends(get_graph_store)],
) -> GraphSchemaResponse:
    """Return the graph schema alongside the counts currently in the database.

    Two audiences. A human gets to see what the graph actually contains without
    opening the Neo4j browser. A reviewer gets to check that the declared schema
    and the loaded data agree - a label with a zero count means ingestion did not
    do what the schema says it should.

    `prompt_schema` is the exact string injected into the agent's routing and
    query-generation prompts, so what the model sees is inspectable too.
    """
    node_counts = {
        row["label"]: row["c"]
        for row in await store.read(
            "MATCH (n) UNWIND labels(n) AS label RETURN label, count(*) AS c"
        )
    }
    rel_counts = {
        row["type"]: row["c"]
        for row in await store.read("MATCH ()-[r]->() RETURN type(r) AS type, count(r) AS c")
    }

    labels = [
        NodeLabelOut(
            label=spec.label,
            description=spec.description,
            properties=list(spec.prompt_properties),
            count=node_counts.get(spec.label, 0),
        )
        for spec in NODE_SPECS
    ]

    relationships = [
        RelationshipOut(
            type=spec.type,
            start=spec.start_label,
            end="|".join(spec.end_label_options) if spec.end_label_field else spec.end_label,
            properties=list(spec.properties),
            description=spec.description,
            count=rel_counts.get(spec.type, 0),
        )
        for spec in REL_SPECS
    ]

    return GraphSchemaResponse(
        labels=labels,
        relationships=relationships,
        # Totals come from the live counts, so Document and Chunk provenance
        # nodes are included even though they are not in the declared schema.
        total_nodes=sum(node_counts.values()),
        total_relationships=sum(rel_counts.values()),
        prompt_schema=schema_summary(),
    )
