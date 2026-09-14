"""The agent graph.

    START -> route
      route --conditional fan-out--> [graph_retrieve] | [semantic_retrieve] | both
      graph_retrieve    --> grade
      semantic_retrieve --> grade
      grade --conditional--> semantic_retrieve   (broaden, once)
                        --> synthesize           (context is sufficient)
                        --> abstain              (nothing usable was found)
      synthesize -> END ;  abstain -> END

**One agent, not several.** The work here is retrieve-then-answer over two
stores. A supervisor delegating to a "graph specialist" and a "vector specialist"
would add message-passing overhead and a second LLM hop to make a decision the
routing node already makes correctly - and it would hide the routing decision,
which is the most interesting thing to show. A single StateGraph with a fan-out
and a grading cycle is the more convincing demonstration because the whole
control flow is visible in one diagram.

**Why the fan-out matters.** Returning a *list* of node names from the routing
edge makes LangGraph run those nodes in the same superstep, concurrently. Their
writes merge through the ``operator.add`` reducers on ``AgentState``. That is
real parallelism, not two sequential calls - which is also why
``LanceDBVectorStore`` pushes its synchronous work to a thread rather than
blocking the loop.
"""

import logging
from typing import Any, Literal, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agrirag.agent.nodes import (
    MAX_RETRIES,
    make_abstain_node,
    make_grade_node,
    make_graph_retrieve_node,
    make_route_node,
    make_semantic_retrieve_node,
    make_synthesize_node,
)
from agrirag.agent.state import AgentResult, AgentState, Route
from agrirag.graph.store import GraphStore
from agrirag.vector.store import VectorStore

logger = logging.getLogger(__name__)

#: The compiled agent. LangGraph's generics are (StateT, ContextT, InputT,
#: OutputT); we use no runtime context and the same TypedDict throughout, so
#: naming the alias once keeps every signature readable.
Agent = CompiledStateGraph[AgentState, None, AgentState, AgentState]


def _fan_out(state: AgentState) -> list[str]:
    """Map the routing decision onto the nodes to run next.

    Returning two names runs both retrievers in one superstep.
    """
    route = state.get("route", "both")
    if route == "graph":
        return ["graph_retrieve"]
    if route == "semantic":
        return ["semantic_retrieve"]
    return ["graph_retrieve", "semantic_retrieve"]


def _after_grade(state: AgentState) -> Literal["semantic_retrieve", "synthesize", "abstain"]:
    """Decide what to do with the graded context.

    The retry edge back into ``semantic_retrieve`` is what makes this a genuine
    cycle rather than a pipeline. It is capped at one pass: a thin result is
    usually a missing fact, and looping on it only burns tokens.
    """
    reason = state.get("sufficiency_reason", "")

    if state.get("sufficient"):
        return "synthesize"

    if state.get("retries", 0) <= MAX_RETRIES and state.get("route") == "graph":
        # A graph-only question that came back thin is the case worth retrying:
        # the corpus may explain what the traversal could not find.
        logger.info("context thin after graph-only retrieval; broadening to semantic")
        return "semantic_retrieve"

    # Retries are exhausted and the grader is unconvinced. The question is now
    # whether anything real was actually retrieved.
    #
    # Abstain only when nothing survived retrieval. That is a deterministic
    # check, and it is the line that matters:
    #
    #   * Graph rows are facts a traversal verified. Refusing to answer while
    #     holding them is wrong - a compound question ("which treatments, and
    #     are any restricted?") often has half its answer in hand, and the
    #     synthesis prompt already instructs the model to say what is missing.
    #   * Chunks have passed MIN_RELEVANCE, so they are genuinely on-topic
    #     rather than nearest-neighbour noise.
    #
    # This branch used to key off `retrieval_context` being non-empty, before
    # the relevance floor existed. Back then it was unsafe: vector search always
    # returns its k nearest neighbours however far away they are, so context was
    # never empty and the agent answered "how do I configure a Kubernetes
    # ingress?" from agronomy chunks. The floor is what fixed that - it empties
    # the context for off-domain questions, so this check reaches `abstain`.
    if not state.get("graph_rows"):
        # No verified graph facts, and the grader - which has seen both the
        # question and the context - says the context cannot answer it. Declining
        # is the honest outcome.
        #
        # Graph rows are the distinction that matters. They are facts a traversal
        # verified, so a compound question ("which treatments, and are any
        # restricted?") that has half its answer in hand should be answered with
        # a caveat rather than refused. Chunks alone that the grader has already
        # rejected are not evidence of anything.
        #
        # This is what the evaluation caught: "what will the wheat price be next
        # season?" scraped one sunn-pest chunk past the relevance floor, and the
        # agent produced an honest "the context does not cover this" answer -
        # correct in substance, but never setting `abstained`, so no metric could
        # see that it had declined.
        logger.info("no graph facts and context judged insufficient: %s", reason)
        return "abstain"

    logger.info("context judged incomplete; answering with what was retrieved: %s", reason)
    return "synthesize"


def build_agent(
    graph_store: GraphStore,
    vector_store: VectorStore,
    checkpointer: Any | None = None,
) -> Agent:
    """Compile the agent.

    ``checkpointer`` enables multi-turn threads via ``thread_id``. It is optional
    so the evaluation suite can run stateless - a checkpointed eval would let one
    golden's state leak into the next.
    """
    builder: StateGraph[AgentState, None, AgentState, AgentState] = StateGraph(AgentState)

    builder.add_node("route", make_route_node())
    builder.add_node("graph_retrieve", make_graph_retrieve_node(graph_store))
    builder.add_node("semantic_retrieve", make_semantic_retrieve_node(vector_store))
    builder.add_node("grade", make_grade_node())
    builder.add_node("synthesize", make_synthesize_node())
    builder.add_node("abstain", make_abstain_node())

    builder.add_edge(START, "route")
    builder.add_conditional_edges("route", _fan_out, ["graph_retrieve", "semantic_retrieve"])
    builder.add_edge("graph_retrieve", "grade")
    builder.add_edge("semantic_retrieve", "grade")
    builder.add_conditional_edges(
        "grade", _after_grade, ["semantic_retrieve", "synthesize", "abstain"]
    )
    builder.add_edge("synthesize", END)
    builder.add_edge("abstain", END)

    return builder.compile(checkpointer=checkpointer)


async def ask(
    agent: Agent,
    question: str,
    *,
    trace_id: str = "unbound",
    thread_id: str | None = None,
    force_route: Route | None = None,
) -> AgentResult:
    """Run one question through the agent and project the result.

    ``force_route`` skips the routing node. It is a demonstration affordance, not
    a production one: forcing 'semantic' on a multi-hop question is how you show
    what the graph traversal contributes.
    """
    config: dict[str, Any] = {"configurable": {"thread_id": thread_id or trace_id}}

    initial: AgentState = {
        "question": question,
        "trace_id": trace_id,
        "forced_route": force_route,
        "retrieval_context": [],
        "graph_rows": [],
        "chunks": [],
        "retries": 0,
    }

    # ainvoke is typed as returning `dict[str, Any] | Any`; the compiled graph
    # always hands back the state dict.
    final = cast("AgentState", await agent.ainvoke(initial, config=config))
    return AgentResult.from_state(final)


def render_mermaid(agent: Agent) -> str:
    """Return the compiled graph as a Mermaid diagram, for the README."""
    return str(agent.get_graph().draw_mermaid())
