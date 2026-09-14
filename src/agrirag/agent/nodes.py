"""The agent's nodes.

Six nodes, each doing one thing:

    route              decide which retrieval path(s) to take
    graph_retrieve     multi-hop traversal over Neo4j
    semantic_retrieve  similarity search over LanceDB
    grade              is the retrieved context enough to answer?
    synthesize         grounded answer with citations
    abstain            honest "I don't know"

Nodes are built by factories that close over the stores, because the stores are
long-lived and created once in the app lifespan. LangGraph nodes take only state,
so closing over dependencies is how they get to a database without constructing
a driver per request.
"""

import logging
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import BaseModel, Field

from agrirag.agent.prompts import (
    ABSTAIN_MESSAGE,
    GRADER_HUMAN,
    GRADER_SYSTEM,
    SYNTHESIS_HUMAN,
    SYNTHESIS_SYSTEM,
    router_system_prompt,
)
from agrirag.agent.state import AgentState, RouteDecision
from agrirag.agent.tools import graph_query, semantic_search
from agrirag.config import get_settings
from agrirag.graph.queries import TEMPLATES_BY_NAME
from agrirag.graph.store import GraphStore
from agrirag.llm.portkey_client import Span, chat_model, structured_model
from agrirag.vector.store import VectorStore

logger = logging.getLogger(__name__)

Node = Callable[[AgentState], Awaitable[dict[str, Any]]]

BASE_K = 5
#: How much wider the retry search goes. One retry only - the point is to show a
#: cycle in the graph, not to build an agent that loops until it finds something.
RETRY_K_BONUS = 5
MAX_RETRIES = 1

#: Similarity floor for a chunk to count as context.
#:
#: Vector search always returns its k nearest neighbours, however far away they
#: are - so without a floor, an off-domain question retrieves five irrelevant
#: chunks, ``retrieval_context`` is non-empty, and the agent answers instead of
#: abstaining. That is the hallucination path this project is meant to avoid.
#:
#: Measured against this corpus, the two populations separate cleanly:
#:   off-domain questions  top score 0.45 - 0.55
#:   in-domain questions   top score 0.78 - 0.84
#: 0.62 sits between them with margin on both sides.
MIN_RELEVANCE = 0.62


def _fit_context(lines: list[str], budget: int) -> list[str]:
    """Trim the context to a character budget, keeping the highest-ranked items.

    Retrieval returns its results best-first, so truncating from the end drops
    the least relevant material. Budgeting in characters rather than item count
    is deliberate: a linearised graph row runs ~200 characters and a corpus chunk
    ~580, so "keep four items" means very different prompt sizes depending on
    which path produced them.
    """
    if budget <= 0:
        return lines

    kept: list[str] = []
    used = 0
    for line in lines:
        if used + len(line) > budget and kept:
            break
        kept.append(line)
        used += len(line)
    return kept


def _context_budget(settings: Any) -> int:
    """Characters of context this provider can actually take."""
    if settings.max_context_chars:
        return int(settings.max_context_chars)
    return 2200 if settings.llm_provider == "ollama" else 0


def _use_llm_grader(settings: Any) -> bool:
    """Should the grading node spend an LLM call on this provider?

    Off by default for small local models. A question costs three LLM calls -
    route, grade, synthesize - and on a CPU-bound 3B model each takes minutes and
    the runner is unstable under that load; dropping to two is the difference
    between completing and failing.

    Abstention survives this. The grader's empty-context branch is deterministic
    and runs regardless, so off-domain questions are still declined. The loss is
    the narrower "on topic but does not answer the question" judgement.
    """
    if settings.llm_grading == "on":
        return True
    if settings.llm_grading == "off":
        return False
    return bool(settings.llm_provider != "ollama")


class Sufficiency(BaseModel):
    """Structured output of the grading node.

    Both fields carry defaults because the model intermittently returns a tool
    call with empty arguments. Defaulting ``sufficient`` to True fails open: when
    we have context, answering it beats refusing because of a transport glitch.
    That is only safe because :data:`MIN_RELEVANCE` has already removed context
    that is merely nearest-neighbour noise - an off-domain question reaches the
    grader with nothing, and the deterministic branch handles it without an LLM
    call at all.
    """

    sufficient: bool = Field(default=True, description="True if the context answers the question.")
    reason: str = Field(default="", description="One sentence explaining the judgement.")


# --------------------------------------------------------------------------
# route
# --------------------------------------------------------------------------


def make_route_node() -> Node:
    """Classify the question into a retrieval route and a graph template."""

    async def route(state: AgentState) -> dict[str, Any]:
        question = state["question"]
        trace_id = state.get("trace_id", "unbound")

        forced = state.get("forced_route")
        if forced:
            # No LLM call: the caller has already decided.
            logger.info("route forced to %s", forced)
            return {
                "route": forced,
                "route_reason": f"route forced to '{forced}' by the caller",
                "template": None,
                "organic_only": False,
            }

        model = structured_model(
            RouteDecision,
            span=Span.ROUTE,
            trace_id=trace_id,
            max_tokens=500,
        )

        try:
            # A provider with a small context window gets the compact prompt.
            compact = _context_budget(get_settings()) > 0
            decision = await model.ainvoke(
                [("system", router_system_prompt(compact=compact)), ("human", question)]
            )
            if not isinstance(decision, RouteDecision):
                # Smaller models sometimes return no tool call at all, and
                # with_structured_output hands back None rather than raising.
                # An `assert` here would be stripped under `python -O`, leaving
                # the None to surface as an AttributeError somewhere less
                # obvious.
                raise TypeError(f"router returned {type(decision).__name__}, not RouteDecision")
        except Exception as exc:
            # Routing is a convenience, not a gate. If the classifier fails,
            # running both retrievers costs one extra query and still answers
            # the question - far better than failing the request outright.
            logger.warning("router failed, defaulting to both: %s", exc)
            return {
                "route": "both",
                "route_reason": f"router unavailable ({type(exc).__name__}); ran both paths",
                "template": None,
                "organic_only": False,
            }

        template = decision.template
        if template and template not in TEMPLATES_BY_NAME:
            # The model named a template that does not exist. Fall back to the
            # deterministic selector in graph_query rather than erroring.
            logger.info("router named unknown template %r; falling back", template)
            template = None

        logger.info("route=%s template=%s (%s)", decision.route, template, decision.reason)
        return {
            "route": decision.route,
            "route_reason": decision.reason,
            "template": template,
            "organic_only": decision.organic_only,
        }

    return route


# --------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------


def make_graph_retrieve_node(store: GraphStore) -> Node:
    """Traverse the knowledge graph."""

    async def graph_retrieve(state: AgentState) -> dict[str, Any]:
        result = await graph_query(
            store,
            state["question"],
            template_name=state.get("template"),
            extra_params={"organic_only": state.get("organic_only", False)},
        )

        if result.error:
            logger.info("graph retrieval found nothing: %s", result.error)

        return {
            "graph_rows": result.rows,
            "retrieval_context": result.context,
            "cypher": result.cypher or None,
            "template": result.template,
            "linked_entities": [
                {"id": e.id, "name": e.name, "label": e.label, "mention": e.mention}
                for e in result.entities
            ],
        }

    return graph_retrieve


def make_semantic_retrieve_node(store: VectorStore) -> Node:
    """Similarity search over the corpus.

    Also the retry target: when grading finds the context thin, control returns
    here with a wider ``k``. Already-seen chunks are filtered out, because the
    reducer appends rather than replaces and a retry would otherwise duplicate
    every hit from the first pass.
    """

    async def semantic_retrieve(state: AgentState) -> dict[str, Any]:
        retries = state.get("retries", 0)
        k = BASE_K + retries * RETRY_K_BONUS

        result = await semantic_search(store, state["question"], k=k)
        if result.error:
            logger.warning("semantic retrieval failed: %s", result.error)
            return {}

        seen = {chunk["chunk_id"] for chunk in state.get("chunks", [])}
        relevant = [hit for hit in result.hits if hit.score >= MIN_RELEVANCE]
        fresh = [hit for hit in relevant if hit.chunk_id not in seen]

        if result.hits and not relevant:
            logger.info(
                "all %d hits below the relevance floor (best %.3f) - treating as no context",
                len(result.hits),
                result.hits[0].score,
            )

        return {
            "chunks": [
                {
                    "chunk_id": hit.chunk_id,
                    "doc_id": hit.doc_id,
                    "doc_title": hit.doc_title,
                    "heading": hit.heading,
                    "score": hit.score,
                }
                for hit in fresh
            ],
            "retrieval_context": [hit.as_context() for hit in fresh],
        }

    return semantic_retrieve


# --------------------------------------------------------------------------
# grade
# --------------------------------------------------------------------------


def make_grade_node() -> Node:
    """Decide whether the retrieved context can answer the question.

    Deterministic check first: no context at all needs no LLM call to judge.
    Only an ambiguous case - context exists but may be off-target - is worth
    spending a request on.
    """

    async def grade(state: AgentState) -> dict[str, Any]:
        context = state.get("retrieval_context", [])
        retries = state.get("retries", 0)

        if not context:
            return {
                "sufficient": False,
                "sufficiency_reason": "no context was retrieved",
                "retries": retries + 1,
            }

        settings = get_settings()
        if not _use_llm_grader(settings):
            # Context exists and the deterministic checks are satisfied. Trusting
            # it is weaker than judging it, but it is what makes the agent
            # complete on a provider that cannot afford a third call.
            logger.info("llm grading disabled for this provider; accepting context")
            return {
                "sufficient": True,
                "sufficiency_reason": "grading skipped (provider has no LLM budget for it)",
                "retries": retries,
            }

        context = _fit_context(context, _context_budget(settings))

        model = structured_model(
            Sufficiency,
            span=Span.GRADE,
            trace_id=state.get("trace_id", "unbound"),
            max_tokens=300,
        )

        try:
            verdict = await model.ainvoke(
                [
                    ("system", GRADER_SYSTEM),
                    (
                        "human",
                        GRADER_HUMAN.format(
                            question=state["question"],
                            context="\n".join(f"- {line}" for line in context),
                        ),
                    ),
                ]
            )
            if not isinstance(verdict, Sufficiency):
                raise TypeError(f"grader returned {type(verdict).__name__}, not Sufficiency")
        except Exception as exc:
            # Failing open: we have context, so answering from it beats refusing.
            logger.warning("grader failed, treating context as sufficient: %s", exc)
            return {
                "sufficient": True,
                "sufficiency_reason": f"grader unavailable ({type(exc).__name__})",
                "retries": retries,
            }

        logger.info("grade: sufficient=%s (%s)", verdict.sufficient, verdict.reason)
        return {
            "sufficient": verdict.sufficient,
            "sufficiency_reason": verdict.reason,
            "retries": retries + (0 if verdict.sufficient else 1),
        }

    return grade


# --------------------------------------------------------------------------
# synthesize / abstain
# --------------------------------------------------------------------------


def _citations(state: AgentState) -> list[dict[str, str]]:
    """Build citations from whatever each retrieval path actually returned."""
    citations: list[dict[str, str]] = []
    seen: set[str] = set()

    for entity in state.get("linked_entities", []):
        if state.get("graph_rows") and entity["id"] not in seen:
            seen.add(entity["id"])
            citations.append({"kind": "graph", "ref": entity["id"], "label": entity["name"]})

    for chunk in state.get("chunks", []):
        if chunk["chunk_id"] in seen:
            continue
        seen.add(chunk["chunk_id"])
        heading = chunk.get("heading") or ""
        label = f"{chunk['doc_title']} - {heading}" if heading else chunk["doc_title"]
        citations.append({"kind": "document", "ref": chunk["chunk_id"], "label": label})

    return citations


def make_synthesize_node() -> Node:
    """Write a grounded answer from the retrieved context."""

    async def synthesize(state: AgentState) -> dict[str, Any]:
        context = _fit_context(state.get("retrieval_context", []), _context_budget(get_settings()))

        model = chat_model(
            span=Span.SYNTHESIZE,
            trace_id=state.get("trace_id", "unbound"),
            max_tokens=900,
        )

        try:
            reply = await model.ainvoke(
                [
                    ("system", SYNTHESIS_SYSTEM),
                    (
                        "human",
                        SYNTHESIS_HUMAN.format(
                            question=state["question"],
                            context="\n".join(f"- {line}" for line in context),
                        ),
                    ),
                ]
            )
        except Exception as exc:
            logger.error("synthesis failed: %s", exc)
            return {
                "answer": ABSTAIN_MESSAGE.format(
                    detail="The language model was unavailable when composing the answer."
                ),
                "abstained": True,
                "citations": [],
            }

        return {
            "answer": str(reply.content).strip(),
            "abstained": False,
            "citations": _citations(state),
        }

    return synthesize


def make_abstain_node() -> Node:
    """Decline honestly rather than synthesising over nothing.

    This node exists so the hallucination metric has a correct-behaviour path to
    reward. Without it, an unanswerable question would still reach the
    synthesiser, which would produce something plausible from an empty context.
    """

    async def abstain(state: AgentState) -> dict[str, Any]:
        reason = state.get("sufficiency_reason", "")
        detail = (
            f"Specifically: {reason}."
            if reason
            else "Nothing in the graph or the document corpus covers it."
        )
        logger.info("abstaining: %s", reason)
        return {
            "answer": ABSTAIN_MESSAGE.format(detail=detail),
            "abstained": True,
            "citations": [],
        }

    return abstain
