"""FastAPI application factory and lifespan wiring."""

import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from agrirag import __version__
from agrirag.agent.graph import build_agent
from agrirag.api.errors import register_error_handlers
from agrirag.api.middleware import RequestIDMiddleware
from agrirag.api.v1 import routes_ask, routes_graph, routes_health, routes_ingest
from agrirag.config import Settings, get_settings
from agrirag.graph.neo4j_store import Neo4jGraphStore
from agrirag.llm.portkey_client import describe_deployment
from agrirag.observability.logging import configure_logging
from agrirag.vector.lancedb_store import LanceDBVectorStore

logger = logging.getLogger(__name__)

DESCRIPTION = """
Agentic **Graph RAG** over an agricultural knowledge graph.

Multi-hop questions are answered by traversing Neo4j; definitional questions by
semantic search over LanceDB. A LangGraph agent decides which path to take, and
declines to answer when neither store has the facts.
"""


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create long-lived clients on startup and release them on shutdown.

    Everything expensive is built once here: the Neo4j driver, the embedded
    vector store, the SQLite checkpointer and the compiled agent graph. Building
    the agent per request would re-read the schema and template catalogue every
    time for no benefit.
    """
    settings: Settings = get_settings()
    configure_logging(settings.log_level)

    async with AsyncExitStack() as stack:
        graph_store = Neo4jGraphStore(
            uri=settings.neo4j_uri,
            user=settings.neo4j_user,
            password=settings.neo4j_password,
            database=settings.neo4j_database,
        )
        stack.push_async_callback(graph_store.close)

        vector_store = LanceDBVectorStore()
        stack.push_async_callback(vector_store.close)

        checkpoint_path = Path(settings.checkpoint_db)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpointer = await stack.enter_async_context(
            AsyncSqliteSaver.from_conn_string(str(checkpoint_path))
        )

        app.state.settings = settings
        app.state.graph_store = graph_store
        app.state.vector_store = vector_store
        app.state.agent = build_agent(graph_store, vector_store, checkpointer=checkpointer)

        logger.info("AgriRAG %s starting (env=%s)", __version__, settings.app_env)
        logger.info("LLM routing: %s", describe_deployment(settings))
        logger.info("Agent threads persisted at %s", checkpoint_path)

        yield

    logger.info("AgriRAG shutdown complete")


def create_app() -> FastAPI:
    """Build the ASGI application."""
    settings = get_settings()

    app = FastAPI(
        title="AgriRAG",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )
    app.add_middleware(RequestIDMiddleware)
    register_error_handlers(app)

    # Root mount for container healthchecks, versioned mount for API consumers.
    app.include_router(routes_health.router, tags=["health"])

    v1 = settings.api_v1_prefix
    app.include_router(routes_health.router, prefix=v1, tags=["health"])
    app.include_router(routes_ask.router, prefix=v1)
    app.include_router(routes_graph.router, prefix=v1)
    app.include_router(routes_ingest.router, prefix=v1)

    return app


app = create_app()
