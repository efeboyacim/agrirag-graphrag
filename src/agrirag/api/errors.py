"""Exception handling.

Two goals: a caller should learn enough to act, and nothing internal should leak.
Every response carries the request id, which is also the Portkey trace id - so a
user reporting "request c0ffee… failed" hands you the exact trace.
"""

import logging

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from neo4j.exceptions import Neo4jError, ServiceUnavailable

from agrirag.api.schemas import ErrorResponse
from agrirag.graph.cypher_guard import UnsafeCypherError
from agrirag.llm.portkey_client import LLMConfigurationError
from agrirag.vector.lancedb_store import VectorStoreNotBuiltError

logger = logging.getLogger(__name__)


def _request_id(request: Request) -> str:
    rid: str = getattr(request.state, "request_id", "unknown")
    return rid


def _error(request: Request, *, detail: str, error_type: str, status_code: int) -> JSONResponse:
    payload = ErrorResponse(detail=detail, error_type=error_type, request_id=_request_id(request))
    return JSONResponse(status_code=status_code, content=payload.model_dump())


def register_error_handlers(app: FastAPI) -> None:
    """Attach handlers that turn infrastructure failures into actionable replies."""

    @app.exception_handler(LLMConfigurationError)
    async def _llm_config(request: Request, exc: LLMConfigurationError) -> JSONResponse:
        # The message from this exception is already written for an operator and
        # contains no secrets, so it is safe to pass through verbatim.
        logger.error("LLM configuration error: %s", exc)
        return _error(
            request,
            detail=str(exc),
            error_type="llm_misconfigured",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    @app.exception_handler(ServiceUnavailable)
    async def _neo4j_down(request: Request, exc: ServiceUnavailable) -> JSONResponse:
        logger.error("Neo4j unavailable: %s", exc)
        return _error(
            request,
            detail="The graph store is not reachable. Check that the neo4j container is running.",
            error_type="graph_unavailable",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    @app.exception_handler(VectorStoreNotBuiltError)
    async def _vectors_missing(request: Request, exc: VectorStoreNotBuiltError) -> JSONResponse:
        logger.error("vector index missing: %s", exc)
        return _error(
            request,
            detail="The vector index has not been built. Run: uv run agrirag-index",
            error_type="vector_index_missing",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    @app.exception_handler(UnsafeCypherError)
    async def _unsafe_cypher(request: Request, exc: UnsafeCypherError) -> JSONResponse:
        # Reaching here means the guard rejected a query - which is the guard
        # working. Log loudly: a committed template tripping it is a real bug.
        logger.error("Cypher guard rejected a query: %s", exc)
        return _error(
            request,
            detail="The generated graph query was rejected by the read-only guard.",
            error_type="unsafe_query",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )

    @app.exception_handler(Neo4jError)
    async def _neo4j_error(request: Request, exc: Neo4jError) -> JSONResponse:
        logger.error("Neo4j error: %s", exc, exc_info=True)
        return _error(
            request,
            detail="The graph query failed.",
            error_type="graph_query_failed",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Deliberately generic: an unexpected exception's message may contain
        # internals. The request id is how it gets tied back to the full
        # traceback in the logs.
        logger.exception("unhandled error on %s %s", request.method, request.url.path)
        return _error(
            request,
            detail="An unexpected error occurred. Quote the request id when reporting it.",
            error_type="internal_error",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )
