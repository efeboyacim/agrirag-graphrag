"""The question endpoint."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends

from agrirag.agent.graph import Agent, ask
from agrirag.api.deps import get_agent, get_request_id
from agrirag.api.schemas import AskRequest, AskResponse, ErrorResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ask"])


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask the agent a question",
    responses={
        503: {"model": ErrorResponse, "description": "A backing store or the gateway is down."},
        500: {"model": ErrorResponse, "description": "Unexpected failure."},
    },
)
async def ask_question(
    payload: AskRequest,
    agent: Annotated[Agent, Depends(get_agent)],
    request_id: Annotated[str, Depends(get_request_id)],
) -> AskResponse:
    """Answer an agricultural question.

    The agent decides whether to traverse the knowledge graph, search the
    document corpus, or both - and declines when neither holds the facts.

    The response returns more than the answer: the route taken and why, the
    Cypher actually executed, the entities the question resolved to, and the
    full retrieval context. That is deliberate. It is what lets a reviewer check
    the answer was grounded rather than take it on trust, and it is exactly what
    the evaluation suite scores.

    `request_id` is the Portkey trace id for this call, so every LLM span raised
    while serving it groups under one trace.

    **Comparing the two paths.** Send the same question twice, once with
    `force_route: "semantic"`. A multi-hop question answered without the graph
    degrades visibly - that contrast is the clearest demonstration of what graph
    retrieval adds.
    """
    logger.info("ask: %r (request_id=%s)", payload.question[:80], request_id)

    result = await ask(
        agent,
        payload.question,
        trace_id=request_id,
        thread_id=payload.thread_id,
        force_route=payload.force_route,
    )

    return AskResponse.from_result(result, request_id)
