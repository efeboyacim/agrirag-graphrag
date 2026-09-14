"""Live tests against the Portkey gateway.

These spend real tokens, so they are few and cheap (32-token replies). They skip
when ``ANTHROPIC_API_KEY`` is unset, which is how CI runs them - the rest of the
integration suite still executes without a key.

What they prove that a unit test cannot: that requests actually traverse the
gateway to Anthropic, that the fallback chain engages, and that the structured
output method chosen in :mod:`agrirag.llm.portkey_client` survives the
OpenAI-to-Anthropic translation.
"""

import json

import httpx
import pytest
from pydantic import BaseModel

from agrirag.config import get_settings
from agrirag.llm.portkey_client import (
    Purpose,
    Span,
    _resolve_config,
    build_headers,
    chat_model,
    structured_model,
)


def _gateway_reachable() -> bool:
    """Is the gateway actually up?

    Checked as well as the API key because these tests need *both*. Without this
    guard, a developer with the stack stopped gets three confusing connection
    failures here while every other integration module skips cleanly - which
    reads as a code regression rather than "Docker is not running".
    """
    root = get_settings().portkey_base_url.removesuffix("/v1").rstrip("/")
    try:
        return httpx.get(root, timeout=3).status_code < 500
    except Exception:
        return False


# Key handling lives in conftest.live_llm, which distinguishes "no key" (skip)
# from "key rejected" (one clear failure) - see tests/integration/conftest.py.
pytestmark = [
    pytest.mark.usefixtures("live_llm"),
    pytest.mark.skipif(
        not _gateway_reachable(),
        reason="Portkey gateway unreachable - run: docker compose up -d gateway",
    ),
]


async def test_a_request_reaches_anthropic_through_the_gateway(reference_llm) -> None:
    reply = await chat_model(span=Span.ROUTE, trace_id="test-live", max_tokens=32).ainvoke(
        "Reply with exactly one word: OK"
    )

    assert reply.content
    assert reply.response_metadata["model_name"] == get_settings().llm_model_primary
    assert reply.usage_metadata["input_tokens"] > 0


async def test_the_fallback_chain_engages_when_the_primary_fails(reference_llm) -> None:
    """Point the primary at a model that does not exist; the reply must come
    from the second target rather than erroring.

    The committed config deliberately does *not* list 404 in ``on_status_codes`` -
    a missing model is a configuration mistake, not the transient overload that
    fallback exists for, and masking it would hide a real bug. The test widens
    the codes to make the mechanism observable on demand.

    Asserted over raw HTTP rather than through ChatOpenAI on purpose. Fallback is
    a property of the gateway, so putting a LangChain client in the middle only
    adds a failure mode: langchain_openai pools httpx clients across instances,
    and pytest-asyncio gives each test its own event loop, so a directly
    constructed client here reuses a connection bound to a loop that has closed.
    """
    settings = get_settings()
    config = _resolve_config(settings, Purpose.APP)
    primary, fallback = (t["override_params"]["model"] for t in config["targets"][:2])

    config["targets"][0]["override_params"]["model"] = "claude-does-not-exist-9"
    config["strategy"]["on_status_codes"] = [400, 404, 429, 500, 502, 503, 529]

    headers = build_headers(settings, span=Span.ROUTE, trace_id="test-fallback")
    headers["x-portkey-config"] = json.dumps(config)
    headers["Authorization"] = f"Bearer {settings.anthropic_api_key}"
    headers["Content-Type"] = "application/json"

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{settings.portkey_base_url}/chat/completions",
            headers=headers,
            json={
                "model": primary,
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "Reply with exactly one word: OK"}],
            },
        )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["model"] == fallback, "the request was not served by the fallback target"
    assert body["choices"][0]["message"]["content"]


async def test_structured_output_survives_the_translation_layer(reference_llm) -> None:
    """`function_calling` is the only method that works through the gateway;
    json_schema and json_mode both fail. Getting this wrong makes extraction
    return nothing for every chunk while reporting success, so it is pinned."""

    class Answer(BaseModel):
        crop: str
        confidence: float

    model = structured_model(Answer, span=Span.EXTRACT, trace_id="test-structured", max_tokens=200)
    result = await model.ainvoke("The Turkish word 'bugday' names which crop? Answer as JSON.")

    assert isinstance(result, Answer)
    assert "wheat" in result.crop.lower()


async def test_the_oss_gateway_does_not_cache() -> None:
    """Pin a measured limitation rather than leave it to prose.

    The committed configs declare ``"cache": {"mode": "simple"}``, and on the
    self-hosted gateway that declaration has no effect - two identical
    back-to-back requests both come back with ``x-portkey-cache-status:
    DISABLED``. Caching needs the hosted control plane's cache store.

    Asserted so the docs cannot drift from reality: if a future gateway release
    starts honouring the config, or someone points this at the hosted plane,
    this test fails and the claim gets revisited instead of quietly rotting.
    """
    settings = get_settings()
    if settings.portkey_api_key:
        pytest.skip("hosted control plane configured - caching is expected to work there")

    headers = build_headers(settings, span=Span.ROUTE, trace_id="test-cache")
    headers["Authorization"] = f"Bearer {settings.anthropic_api_key}"
    headers["Content-Type"] = "application/json"
    payload = {
        "model": settings.llm_model_primary,
        "max_tokens": 24,
        "messages": [{"role": "user", "content": "Reply with exactly: CACHEPROBE"}],
    }

    statuses = []
    async with httpx.AsyncClient(timeout=60) as client:
        for _ in range(2):
            response = await client.post(
                f"{settings.portkey_base_url}/chat/completions", headers=headers, json=payload
            )
            assert response.status_code == 200, response.text
            statuses.append(response.headers.get("x-portkey-cache-status"))

    assert statuses == ["DISABLED", "DISABLED"], (
        f"gateway reported cache statuses {statuses} - if caching now works on the "
        "OSS gateway, update docs/portkey.md and portkey/README.md"
    )
