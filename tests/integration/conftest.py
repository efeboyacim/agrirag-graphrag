"""Shared setup for integration tests.

The point of this module is to distinguish three states that otherwise all look
like "the tests are broken":

    no key configured        -> skip, cleanly. This is how CI runs on a fork PR.
    key configured, usable   -> run.
    key rejected (401/403)   -> fail once: the credential is wrong.
    account out of credit    -> fail once: the credential is fine, the balance is not.

The last two are why this exists, and they are deliberately distinguished because
the remedy differs - rotate a key versus top up a balance. Both otherwise surface
as a pile of opaque ``OpenAIConnectionError`` traces that read like a code
regression.

Both were hit for real while building this: a key was revoked mid-run, and a
replacement key turned out to be valid but attached to an account with no credit.
The second returns HTTP 400, not 401, so a guard that only checks for auth codes
waves it straight through.
"""

import httpx
import pytest

from agrirag.config import get_settings

_probe_result: bool | None = None
_reason: str = ""


def _ollama_is_up(settings: object) -> bool:
    """Is the configured local model reachable and pulled?"""
    host = settings.ollama_host.replace("host.docker.internal", "localhost")  # type: ignore[attr-defined]
    try:
        tags = httpx.get(f"{host}/api/tags", timeout=8)
    except Exception:
        return False
    if tags.status_code != 200:
        return False
    wanted = settings.ollama_model  # type: ignore[attr-defined]
    names = {m.get("name", "") for m in tags.json().get("models", [])}
    return any(n == wanted or n.split(":")[0] == wanted.split(":")[0] for n in names)


def _key_is_accepted() -> bool:
    """Ask the active provider directly whether it will serve a request.

    Deliberately bypasses the gateway. If this goes through the gateway a
    rejection is ambiguous - gateway misconfiguration and a bad credential look
    the same - and the whole purpose here is to name the cause precisely.

    Cached for the session: one probe, not one per test.
    """
    global _probe_result, _reason
    if _probe_result is not None:
        return _probe_result

    settings = get_settings()

    # Probe whatever is actually configured. Checking Anthropic while the system
    # is pointed at a local model would report a credential problem that has no
    # bearing on the run.
    if settings.llm_provider == "ollama":
        if _ollama_is_up(settings):
            _probe_result = True
        else:
            _reason = (
                f"LLM_PROVIDER=ollama but {settings.ollama_model} is not available at "
                f"{settings.ollama_host}. Start Ollama and run: "
                f"ollama pull {settings.ollama_model}."
            )
            _probe_result = False
        return _probe_result

    if settings.llm_provider == "groq":
        try:
            response = httpx.get(
                "https://api.groq.com/openai/v1/models",
                headers={"Authorization": f"Bearer {settings.groq_api_key}"},
                timeout=20,
            )
            if response.status_code in (401, 403):
                _reason = (
                    f"Groq rejected the key (HTTP {response.status_code}). It has most "
                    "likely been revoked or mistyped. Update GROQ_API_KEY in .env."
                )
                _probe_result = False
                return _probe_result
            _probe_result = True
        except Exception:
            # Network trouble, not a credential problem - see the anthropic
            # branch below for the same reasoning.
            _probe_result = True
        return _probe_result

    try:
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": settings.anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": settings.llm_model_fallback,
                "max_tokens": 4,
                "messages": [{"role": "user", "content": "hi"}],
            },
            timeout=20,
        )
        if response.status_code in (401, 403):
            _reason = (
                f"Anthropic rejected the key (HTTP {response.status_code}). It has most "
                "likely been revoked, rotated or expired. Update ANTHROPIC_API_KEY in .env."
            )
            _probe_result = False
            return _probe_result

        # An exhausted balance comes back as 400, not 401, so a check that only
        # looks at auth codes lets it through and the tests then fail opaquely.
        if response.status_code == 400 and "credit balance" in response.text.lower():
            _reason = (
                "The key is valid but the Anthropic account has insufficient credit. "
                "Add credit under Plans & Billing; no code or key change is needed."
            )
            _probe_result = False
            return _probe_result

        # Anything else - including a rate limit - means the credential works.
        _probe_result = True
    except Exception:
        # Network trouble is not a credential problem; let the tests run and
        # fail on their own terms rather than masking a real outage as "bad key".
        _probe_result = True

    return _probe_result


@pytest.fixture(scope="session")
def live_llm() -> None:
    """Require a working Anthropic credential, or explain precisely what is wrong."""
    settings = get_settings()

    if settings.llm_provider == "anthropic" and not settings.anthropic_api_key:
        pytest.skip("ANTHROPIC_API_KEY not set - live LLM tests skipped")
    if settings.llm_provider == "groq" and not settings.groq_api_key:
        pytest.skip("GROQ_API_KEY not set - live LLM tests skipped")

    if not _key_is_accepted():
        pytest.fail(
            f"{_reason} This is an account problem, not a failure of the code under "
            "test - everything that does not need a model still passes.",
            pytrace=False,
        )


@pytest.fixture(scope="session")
def reference_llm(live_llm: None) -> None:
    """Require the reference provider (Anthropic), not merely a working one.

    Two kinds of test need this, and both are legitimate:

    * Tests asserting Anthropic-specific configuration - that a reply came from
      ``claude-sonnet-5``, or that the two-target fallback chain engaged. The
      local config has one target and a different model, so these assert things
      that are simply not true of it.
    * Tests asserting answer *content* - that the answer names spinosad, or lists
      three of four rotation partners. Those measure model quality, and a 3B
      local model does not reach it.

    Everything structural - routing, retrieval counts, abstention, node sequence,
    error handling - stays provider-agnostic and runs against whatever is
    configured. That split is the point: the system is testable on a free local
    model, while the claims that only hold for the reference model stay honest
    about it.
    """
    if get_settings().llm_provider != "anthropic":
        pytest.skip(
            f"LLM_PROVIDER={get_settings().llm_provider} - this test asserts "
            "reference-provider behaviour or answer quality"
        )
