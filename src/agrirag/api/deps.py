"""FastAPI dependency providers.

Long-lived clients are created once in the app lifespan and handed out from
``app.state``, so nothing constructs a driver per request.
"""

from fastapi import Request

from agrirag.agent.graph import Agent
from agrirag.graph.store import GraphStore
from agrirag.vector.store import VectorStore


def get_graph_store(request: Request) -> GraphStore:
    """Return the process-wide graph store."""
    store: GraphStore = request.app.state.graph_store
    return store


def get_vector_store(request: Request) -> VectorStore:
    """Return the process-wide vector store."""
    store: VectorStore = request.app.state.vector_store
    return store


def get_agent(request: Request) -> Agent:
    """Return the compiled agent.

    Compiled once at startup: building the graph re-reads the schema and the
    template catalogue, which is wasted work on every request.
    """
    agent: Agent = request.app.state.agent
    return agent


def get_request_id(request: Request) -> str:
    """Return this request's correlation id.

    Phase 2 forwards this to Portkey as ``trace_id`` so that every LLM span
    raised while serving one API call groups under a single trace.
    """
    request_id: str = getattr(request.state, "request_id", "unknown")
    return request_id
