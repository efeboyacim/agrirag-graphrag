"""The only place in the codebase that constructs an LLM client.

Router, Cypher generation, entity extraction, context grading, answer synthesis
and the Phase 5 evaluation judge all call :func:`chat_model` here. That single
chokepoint is what makes "every LLM call routes through Portkey" a verifiable
property rather than a claim - there is no second code path that could bypass it,
and :mod:`tests.unit.test_portkey_client` asserts it.

Two deployment modes, selected by whether ``PORTKEY_API_KEY`` is set:

* **Self-hosted OSS gateway** (default). The config JSON is sent inline on every
  request in the ``x-portkey-config`` header. No account needed.
* **Hosted control plane.** The same JSON is saved as a config in Portkey and
  referenced by id. Set ``PORTKEY_API_KEY`` and ``PORTKEY_CONFIG_APP`` /
  ``PORTKEY_CONFIG_EVAL``.

Switching between them is configuration, not code.
"""

import json
import logging
from copy import deepcopy
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from agrirag.config import Settings, get_settings

logger = logging.getLogger(__name__)

PORTKEY_DIR = Path(__file__).resolve().parents[3] / "portkey"


class Purpose(StrEnum):
    """Which Portkey config a call should use.

    Keeping evaluation traffic on its own config is what lets the dashboard
    separate "what the API cost" from "what the nightly eval run cost".
    """

    APP = "app"
    EVAL = "eval"


class Span(StrEnum):
    """Logical step a request belongs to.

    Sent as Portkey metadata so a trace can be read step by step rather than as
    an undifferentiated pile of completions.
    """

    ROUTE = "route"
    CYPHER_GEN = "cypher_gen"
    EXTRACT = "extract"
    GRADE = "grade"
    SYNTHESIZE = "synthesize"
    EVAL_JUDGE = "eval_judge"


class LLMConfigurationError(RuntimeError):
    """Raised when the gateway cannot be addressed with the current settings."""


@lru_cache
def load_config(purpose: Purpose, provider: str = "anthropic") -> dict[str, Any]:
    """Read a committed Portkey config file.

    Provider selection is a *file* choice, not a code branch: switching to a
    local model loads ``config.app.ollama.json`` instead of ``config.app.json``
    and nothing else changes. That is the claim the gateway exists to make good
    on - adding or swapping a provider should not touch Python.
    """
    suffix = "" if provider == "anthropic" else f".{provider}"
    path = PORTKEY_DIR / f"config.{purpose.value}{suffix}.json"
    if not path.exists():
        raise LLMConfigurationError(f"Portkey config not found: {path}")
    parsed: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return parsed


def _resolve_config(settings: Settings, purpose: Purpose) -> dict[str, Any]:
    """Return the config with the provider credential injected into each target.

    The committed config files hold no secrets - the key is added here, at request
    time. On the hosted plane this is what a virtual key would do; the OSS gateway
    has no credential store, so each target carries ``api_key`` directly.
    """
    config = deepcopy(load_config(purpose, settings.llm_provider))

    if settings.llm_provider == "ollama":
        # A local model needs no credential; it needs an address.
        for target in config.get("targets", []):
            target.setdefault("custom_host", settings.ollama_host)
            target.setdefault("override_params", {}).setdefault("model", settings.ollama_model)
        return config

    for target in config.get("targets", []):
        target.setdefault("api_key", settings.anthropic_api_key)
    return config


def _config_headers(settings: Settings, purpose: Purpose) -> dict[str, str]:
    """Tell the gateway which config to apply.

    The hosted plane resolves a config id; the OSS gateway has no config store,
    so the whole JSON travels in the header instead.
    """
    if settings.portkey_api_key:
        config_id = (
            settings.portkey_config_app if purpose is Purpose.APP else settings.portkey_config_eval
        )
        if not config_id:
            raise LLMConfigurationError(
                f"PORTKEY_API_KEY is set but the config id for '{purpose.value}' is empty. "
                f"Set PORTKEY_CONFIG_{purpose.value.upper()}, or clear PORTKEY_API_KEY to use "
                "the self-hosted gateway."
            )
        return {"x-portkey-config": config_id}

    return {"x-portkey-config": json.dumps(_resolve_config(settings, purpose))}


def build_headers(
    settings: Settings,
    *,
    purpose: Purpose = Purpose.APP,
    span: Span,
    trace_id: str,
    metadata: dict[str, str] | None = None,
) -> dict[str, str]:
    """Assemble the Portkey headers for one call.

    ``trace_id`` is the FastAPI request id, so every LLM span raised while serving
    one API call groups under a single trace. ``span`` names the step. Together
    they turn the trace view into something readable during a demo.
    """
    if settings.llm_provider == "anthropic" and not settings.anthropic_api_key:
        raise LLMConfigurationError(
            "ANTHROPIC_API_KEY is not set. Add it to .env - the gateway forwards it "
            "to the provider and holds no credentials of its own. "
            "Alternatively set LLM_PROVIDER=ollama to run against a local model."
        )

    headers = {
        "x-portkey-provider": settings.llm_provider,
        "x-portkey-metadata": json.dumps(
            {
                "_user": "agrirag",
                "trace_id": trace_id,
                "span": span.value,
                "purpose": purpose.value,
                **(metadata or {}),
            }
        ),
        "x-portkey-trace-id": trace_id,
        **_config_headers(settings, purpose),
    }

    if settings.portkey_api_key:
        headers["x-portkey-api-key"] = settings.portkey_api_key

    return headers


def chat_model(
    *,
    span: Span,
    trace_id: str = "unbound",
    purpose: Purpose = Purpose.APP,
    temperature: float | None = None,
    max_tokens: int = 1024,
    metadata: dict[str, str] | None = None,
    settings: Settings | None = None,
) -> ChatOpenAI:
    """Build a LangChain chat model that speaks to the Portkey gateway.

    ``ChatOpenAI`` rather than ``ChatAnthropic`` is not an accident: the gateway
    exposes an OpenAI-compatible surface and performs the translation to the
    Anthropic API itself. Pointing a native Anthropic client at api.anthropic.com
    would bypass the gateway entirely and silently lose fallback, caching and
    tracing - which is exactly the failure this module exists to prevent.

    The model name is set by the Portkey config's targets, so ``model`` here is a
    placeholder the gateway overrides via ``override_params``.

    ``temperature`` defaults to None and is then omitted from the payload
    entirely. The Claude 5 models reject it as deprecated, and these calls are
    structured extraction and classification where sampling control would not
    help anyway.
    """
    settings = settings or get_settings()

    optional: dict[str, Any] = {}
    if temperature is not None:
        optional["temperature"] = temperature

    model_name = (
        settings.ollama_model if settings.llm_provider == "ollama" else settings.llm_model_primary
    )
    timeout = settings.llm_timeout_seconds or (300 if settings.llm_provider == "ollama" else 60)

    return ChatOpenAI(
        model=model_name,
        base_url=settings.portkey_base_url,
        api_key=settings.anthropic_api_key or "local",  # type: ignore[arg-type]
        default_headers=build_headers(
            settings,
            purpose=purpose,
            span=span,
            trace_id=trace_id,
            metadata=metadata,
        ),
        max_completion_tokens=max_tokens,
        timeout=timeout,
        max_retries=0,  # the gateway owns retry policy; double-retrying hides failures
        **optional,
    )


def structured_model(
    schema: type[BaseModel],
    *,
    span: Span,
    trace_id: str = "unbound",
    purpose: Purpose = Purpose.APP,
    max_tokens: int = 1024,
    metadata: dict[str, str] | None = None,
    settings: Settings | None = None,
) -> Runnable[Any, BaseModel]:
    """Build a model that returns ``schema`` instances.

    **Always ``method="function_calling"``.** This is a gateway-specific
    constraint discovered by testing, not a style preference:

    * ``function_calling`` works - Portkey translates OpenAI tool calls into
      Anthropic's native tool-use API, and the response carries a real
      ``toolu_`` tool call.
    * ``json_schema`` fails - the translated response has no ``parsed`` field.
    * ``json_mode`` fails - the model replies with fenced markdown, sometimes
      YAML, and the parser rejects it.

    The failure mode of getting this wrong is nasty: extraction silently returns
    nothing for every chunk while the pipeline reports success. Centralising the
    choice here means no caller can reintroduce it.
    """
    model = chat_model(
        span=span,
        trace_id=trace_id,
        purpose=purpose,
        max_tokens=max_tokens,
        metadata=metadata,
        settings=settings,
    )
    # with_structured_output is typed as returning dict | BaseModel because the
    # dict form is possible for other methods; function_calling always yields
    # the model instance.
    return cast(
        "Runnable[Any, BaseModel]",
        model.with_structured_output(schema, method="function_calling"),
    )


def describe_deployment(settings: Settings | None = None) -> str:
    """One-line summary of how LLM traffic is currently routed, for logs and /health."""
    settings = settings or get_settings()
    mode = "hosted control plane" if settings.portkey_api_key else "self-hosted OSS gateway"
    target = (
        f"{settings.llm_provider}/{settings.ollama_model}"
        if settings.llm_provider == "ollama"
        else f"anthropic/{settings.llm_model_primary}"
    )
    return f"Portkey {mode} at {settings.portkey_base_url} -> {target}"
