"""The two retrieval tools.

Defined as LangChain ``@tool`` functions so they are independently testable and
bindable to a model, but the Phase 3 agent invokes them from explicit graph nodes
rather than through a ReAct ``ToolNode``. Explicit invocation keeps the control
flow deterministic and makes retrieval failures attributable to a specific node -
a ReAct loop would bury the routing decision inside model tool-calling and make
the Phase 5 retrieval metrics much harder to interpret.

Both tools return their findings as ``context``: a list of natural-language
sentences. Graph rows are linearised through their template's sentence format.
This is the mechanism that lets DeepEval's contextual metrics score graph facts
and text chunks with the same metric.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from agrirag.graph.cypher_guard import UnsafeCypherError, validate
from agrirag.graph.entities import EntityRef, link_entities, to_params
from agrirag.graph.queries import TEMPLATES, TEMPLATES_BY_NAME, QueryTemplate, bind_params
from agrirag.graph.store import GraphStore
from agrirag.vector.store import SearchHit, VectorStore

logger = logging.getLogger(__name__)

DEFAULT_K = 5
DEFAULT_ROW_LIMIT = 25


@dataclass
class GraphResult:
    """What ``graph_query`` found, plus everything needed to explain it."""

    context: list[str] = field(default_factory=list)
    rows: list[dict[str, Any]] = field(default_factory=list)
    cypher: str = ""
    template: str | None = None
    params: dict[str, Any] = field(default_factory=dict)
    entities: list[EntityRef] = field(default_factory=list)
    error: str | None = None

    @property
    def found(self) -> bool:
        return bool(self.rows)


@dataclass
class SemanticResult:
    """What ``semantic_search`` found."""

    context: list[str] = field(default_factory=list)
    hits: list[SearchHit] = field(default_factory=list)
    error: str | None = None

    @property
    def found(self) -> bool:
        return bool(self.hits)


def select_template(params: dict[str, str]) -> QueryTemplate | None:
    """Pick the first template whose required parameters are all available.

    A deterministic default so the tool is usable on its own. Phase 3 replaces
    this with an LLM choosing from :func:`~agrirag.graph.queries.template_catalogue`,
    which handles intent - this fallback only handles availability.

    A template must bind at least one real entity to be selectable. Without that
    rule, a template with no required parameters matches *every* question,
    including off-domain ones, and returns a full table dump as though it were an
    answer. Retrieving nothing is the correct outcome for "how do I configure a
    Kubernetes ingress?" - it is what lets the agent abstain instead of
    synthesising over irrelevant rows.
    """
    if not any(params.values()):
        return None

    for template in TEMPLATES:
        if not all(params.get(p) for p in template.required_params):
            continue
        if not template.required_params and not any(
            params.get(p) for p in template.optional_params
        ):
            continue
        return template
    return None


async def graph_query(
    store: GraphStore,
    question: str,
    *,
    template_name: str | None = None,
    extra_params: dict[str, Any] | None = None,
    row_limit: int = DEFAULT_ROW_LIMIT,
) -> GraphResult:
    """Answer a question by traversing the knowledge graph.

    Steps, in order:

    1. Resolve entity mentions to canonical node ids via the fulltext index.
    2. Select a parameterised template (or use the one named).
    3. Bind parameters, validate through the Cypher guard, execute read-only.
    4. Linearise rows into sentences.
    """
    entities = await link_entities(store, question)
    params: dict[str, Any] = to_params(entities)
    if extra_params:
        params.update({k: v for k, v in extra_params.items() if v is not None})

    template = TEMPLATES_BY_NAME.get(template_name) if template_name else select_template(params)
    if template is None:
        reason = (
            f"unknown template '{template_name}'"
            if template_name
            else "no template matched the entities resolved from the question"
        )
        logger.info("graph_query: %s (entities=%s)", reason, [e.id for e in entities])
        return GraphResult(entities=entities, error=reason)

    try:
        bound = bind_params(template, params)
    except ValueError as exc:
        return GraphResult(entities=entities, template=template.name, error=str(exc))

    try:
        guarded = validate(template.cypher, default_limit=row_limit)
    except UnsafeCypherError as exc:  # pragma: no cover - templates are committed
        logger.error("template %s failed the Cypher guard: %s", template.name, exc)
        return GraphResult(entities=entities, template=template.name, error=str(exc))

    rows = await store.read(guarded.cypher, bound)

    return GraphResult(
        context=[template.render(row) for row in rows],
        rows=rows,
        cypher=guarded.cypher.strip(),
        template=template.name,
        params=bound,
        entities=entities,
    )


async def semantic_search(
    store: VectorStore,
    query: str,
    *,
    k: int = DEFAULT_K,
) -> SemanticResult:
    """Answer a question by similarity search over the chunked corpus."""
    try:
        hits = await store.search(query, k=k)
    except Exception as exc:
        logger.warning("semantic search failed: %s", exc)
        return SemanticResult(error=str(exc))

    return SemanticResult(context=[hit.as_context() for hit in hits], hits=hits)


# --------------------------------------------------------------------------
# LangChain tool wrappers.
#
# The stores are long-lived and created in the app lifespan, so the tools are
# built by factories that close over them rather than constructing a driver per
# call.
# --------------------------------------------------------------------------


def make_graph_query_tool(store: GraphStore) -> Any:
    """Build the ``graph_query_tool`` bound to a graph store."""
    from langchain_core.tools import tool

    @tool("graph_query_tool")
    async def _graph_query_tool(question: str, template_name: str | None = None) -> str:
        """Query the agricultural knowledge graph by traversing relationships.

        Use for questions that depend on how entities relate to each other: which
        pests attack a crop, what treats them, what is restricted where, which
        practices suit a soil, what rotates with what. Handles multi-hop
        questions that similarity search cannot answer.
        """
        result = await graph_query(store, question, template_name=template_name)
        if result.error:
            return f"No graph results: {result.error}"
        return "\n".join(result.context) or "No matching facts in the graph."

    return _graph_query_tool


def make_semantic_search_tool(store: VectorStore) -> Any:
    """Build the ``semantic_search_tool`` bound to a vector store."""
    from langchain_core.tools import tool

    @tool("semantic_search_tool")
    async def _semantic_search_tool(query: str, k: int = DEFAULT_K) -> str:
        """Search the agronomy document corpus by semantic similarity.

        Use for definitional, explanatory and how-does-it-work questions -
        concepts, mechanisms, background - where the answer is prose rather than
        a relationship between named entities.
        """
        result = await semantic_search(store, query, k=k)
        if result.error:
            return f"Semantic search unavailable: {result.error}"
        return "\n\n".join(result.context) or "No relevant passages found."

    return _semantic_search_tool
