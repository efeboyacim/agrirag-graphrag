"""Health endpoint.

Mounted twice: at ``/health`` for container orchestration and under the
versioned prefix for API consumers.

Always returns 200. The container is alive even when a dependency is not, and a
non-200 here would make Docker restart a process whose own code is fine. Callers
branch on the ``status`` field, not the HTTP code.
"""

import logging
from typing import Annotated

import httpx
from fastapi import APIRouter, Depends

from agrirag import __version__
from agrirag.api.deps import get_graph_store, get_vector_store
from agrirag.api.schemas import DependencyStatus, HealthResponse
from agrirag.config import get_settings
from agrirag.graph.store import GraphStore
from agrirag.llm.portkey_client import describe_deployment
from agrirag.vector.store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter()


async def _gateway_status() -> DependencyStatus:
    """Check the gateway is reachable without spending a token.

    Deliberately does not send a completion: a health check that costs money on
    every probe is a health check people disable.
    """
    settings = get_settings()
    root = settings.portkey_base_url.removesuffix("/v1").rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            response = await client.get(root)
        return "ok" if response.status_code < 500 else "unavailable"
    except Exception:
        logger.warning("gateway check failed at %s", root, exc_info=True)
        return "unavailable"


@router.get("/health", response_model=HealthResponse, summary="Liveness and dependency readiness")
async def health(
    graph: Annotated[GraphStore, Depends(get_graph_store)],
    vectors: Annotated[VectorStore, Depends(get_vector_store)],
) -> HealthResponse:
    """Report app liveness and whether each backing store is reachable."""
    neo4j_status: DependencyStatus = "ok"
    try:
        await graph.verify_connectivity()
    except Exception:
        logger.warning("Neo4j connectivity check failed", exc_info=True)
        neo4j_status = "unavailable"

    lancedb_status: DependencyStatus = "ok"
    chunks = 0
    try:
        chunks = await vectors.count()
    except Exception:
        logger.warning("LanceDB check failed", exc_info=True)
        lancedb_status = "unavailable"

    gateway_status = await _gateway_status()

    everything_ok = all(status == "ok" for status in (neo4j_status, lancedb_status, gateway_status))

    return HealthResponse(
        status="ok" if everything_ok else "degraded",
        version=__version__,
        neo4j=neo4j_status,
        lancedb=lancedb_status,
        gateway=gateway_status,
        chunks=chunks,
        llm_routing=describe_deployment(),
    )
