"""LLM gateway wiring.

The load-bearing test here is :func:`test_no_module_constructs_an_llm_outside_the_factory`.
"Everything routes through Portkey" is only true if nothing else can build a
client, and that is a property of the source tree rather than of any single
function - so it is checked by scanning the source tree.
"""

import json
from pathlib import Path

import pytest

from agrirag.config import Settings
from agrirag.llm.portkey_client import (
    LLMConfigurationError,
    Purpose,
    Span,
    _resolve_config,
    build_headers,
    chat_model,
    describe_deployment,
    load_config,
)

SRC = Path("src/agrirag")


def _settings(**overrides: object) -> Settings:
    """Build settings for a test, isolated from the developer's .env.

    ``_env_file=None`` matters: without it pydantic-settings reads the real .env,
    so a local `LLM_PROVIDER=ollama` silently changes what these tests exercise.
    That happened - two tests started failing purely because the machine was
    pointed at a local model, which is the kind of failure that wastes an
    afternoon looking for a bug in the code under test.
    """
    base: dict[str, object] = {
        "llm_provider": "anthropic",
        "anthropic_api_key": "sk-ant-test",
        "portkey_base_url": "http://gateway:8787/v1",
        "portkey_api_key": "",
        "portkey_config_app": "",
        "portkey_config_eval": "",
    }
    base.update(overrides)
    return Settings(_env_file=None, **base)  # type: ignore[arg-type,call-arg]


# --------------------------------------------------------------------------
# The chokepoint property
# --------------------------------------------------------------------------


def test_no_module_constructs_an_llm_outside_the_factory() -> None:
    """Nothing outside llm/ may import a provider SDK or chat class directly.

    This is what makes the gateway unavoidable. A developer reaching for
    ChatAnthropic to "just quickly test something" would silently bypass
    fallback, caching and tracing - and nothing else in the suite would notice.
    """
    forbidden = ("langchain_openai", "langchain_anthropic", "import anthropic", "from anthropic")
    offenders: list[str] = []

    for path in SRC.rglob("*.py"):
        if path.parts[-2] == "llm":
            continue
        source = path.read_text(encoding="utf-8")
        for needle in forbidden:
            if needle in source:
                offenders.append(f"{path}: {needle}")

    assert offenders == [], "these modules bypass agrirag.llm.portkey_client:\n  " + "\n  ".join(
        offenders
    )


def test_nothing_outside_the_factory_names_the_provider_endpoint() -> None:
    """A hard-coded api.anthropic.com anywhere would route around the gateway.

    llm/ is exempt: the factory's docstring names the endpoint precisely to
    explain why no one should call it directly.
    """
    offenders = [
        str(path)
        for path in SRC.rglob("*.py")
        if path.parts[-2] != "llm" and "api.anthropic.com" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@pytest.mark.parametrize("purpose", list(Purpose))
def test_both_configs_exist_and_declare_anthropic_targets(purpose: Purpose) -> None:
    config = load_config(purpose)

    assert config["targets"], f"{purpose} has no targets"
    assert all(t["provider"] == "anthropic" for t in config["targets"])


def test_the_app_config_declares_a_fallback_chain() -> None:
    config = load_config(Purpose.APP)

    assert config["strategy"]["mode"] == "fallback"
    assert len(config["targets"]) >= 2, "a fallback chain needs a second target"
    models = [t["override_params"]["model"] for t in config["targets"]]
    assert len(set(models)) == len(models), "fallback targets must differ"


def test_the_eval_config_has_no_fallback() -> None:
    """A judge that silently degrades to a smaller model makes scores
    incomparable between runs."""
    assert len(load_config(Purpose.EVAL)["targets"]) == 1


def test_committed_configs_contain_no_credentials() -> None:
    for purpose in Purpose:
        raw = json.dumps(load_config(purpose))
        assert "sk-ant" not in raw
        assert "api_key" not in raw


def test_the_credential_is_injected_at_request_time() -> None:
    resolved = _resolve_config(_settings(), Purpose.APP)

    assert all(t["api_key"] == "sk-ant-test" for t in resolved["targets"])
    # ...and the committed file is unchanged, i.e. the injection is on a copy.
    assert "api_key" not in json.dumps(load_config(Purpose.APP))


def test_groq_gets_a_credential_and_its_configured_model() -> None:
    """Groq is hosted like Anthropic (a credential, no address) but the model
    still comes from Settings, like ollama - the free tier's lineup is the
    part most likely to need changing without a code change."""
    resolved = _resolve_config(
        _settings(llm_provider="groq", groq_api_key="gsk-test", groq_model="llama-3.1-8b-instant"),
        Purpose.APP,
    )

    assert all(t["api_key"] == "gsk-test" for t in resolved["targets"])
    # The committed fallback chain already names two distinct models; setdefault
    # must not clobber them with the single configured groq_model.
    models = [t["override_params"]["model"] for t in resolved["targets"]]
    assert len(set(models)) == len(models)


def test_groq_configs_declare_a_fallback_chain_and_no_credentials() -> None:
    app = load_config(Purpose.APP, provider="groq")
    assert app["strategy"]["mode"] == "fallback"
    assert len(app["targets"]) >= 2
    assert all(t["provider"] == "groq" for t in app["targets"])
    assert "api_key" not in json.dumps(app)

    eval_ = load_config(Purpose.EVAL, provider="groq")
    assert len(eval_["targets"]) == 1, "a degrading judge makes scores incomparable"


# --------------------------------------------------------------------------
# Headers
# --------------------------------------------------------------------------


def test_headers_carry_the_trace_id_and_span() -> None:
    headers = build_headers(_settings(), span=Span.SYNTHESIZE, trace_id="req-42")

    assert headers["x-portkey-trace-id"] == "req-42"
    metadata = json.loads(headers["x-portkey-metadata"])
    assert metadata["span"] == "synthesize"
    assert metadata["trace_id"] == "req-42"


def test_extra_metadata_is_merged() -> None:
    headers = build_headers(
        _settings(), span=Span.EXTRACT, trace_id="t", metadata={"chunk_id": "chunk_1"}
    )

    assert json.loads(headers["x-portkey-metadata"])["chunk_id"] == "chunk_1"


def test_the_oss_gateway_receives_the_config_inline() -> None:
    headers = build_headers(_settings(), span=Span.ROUTE, trace_id="t")

    config = json.loads(headers["x-portkey-config"])
    assert config["strategy"]["mode"] == "fallback"
    assert "x-portkey-api-key" not in headers


def test_the_hosted_plane_receives_a_config_id() -> None:
    headers = build_headers(
        _settings(portkey_api_key="pk-test", portkey_config_app="pc-abc"),
        span=Span.ROUTE,
        trace_id="t",
    )

    assert headers["x-portkey-config"] == "pc-abc"
    assert headers["x-portkey-api-key"] == "pk-test"


def test_a_hosted_key_without_a_config_id_fails_loudly() -> None:
    """Silently falling back to inline config would send the credential to a
    plane that does not expect it."""
    with pytest.raises(LLMConfigurationError, match="PORTKEY_CONFIG_APP"):
        build_headers(_settings(portkey_api_key="pk-test"), span=Span.ROUTE, trace_id="t")


def test_a_missing_provider_key_fails_with_an_actionable_message() -> None:
    with pytest.raises(LLMConfigurationError, match="ANTHROPIC_API_KEY"):
        build_headers(_settings(anthropic_api_key=""), span=Span.ROUTE, trace_id="t")


def test_a_missing_groq_key_fails_with_an_actionable_message() -> None:
    with pytest.raises(LLMConfigurationError, match="GROQ_API_KEY"):
        build_headers(_settings(llm_provider="groq"), span=Span.ROUTE, trace_id="t")


# --------------------------------------------------------------------------
# Model construction
# --------------------------------------------------------------------------


def test_the_model_points_at_the_gateway_not_the_provider() -> None:
    model = chat_model(span=Span.ROUTE, trace_id="t", settings=_settings())

    assert str(model.openai_api_base) == "http://gateway:8787/v1"


def test_temperature_is_omitted_by_default() -> None:
    """The Claude 5 models reject `temperature` as deprecated, so it must not be
    sent unless a caller explicitly asks for it."""
    assert chat_model(span=Span.ROUTE, settings=_settings()).temperature is None
    assert chat_model(span=Span.ROUTE, temperature=0.5, settings=_settings()).temperature == 0.5


def test_groq_model_selection_and_timeout() -> None:
    """Groq shares Anthropic's timeout, not Ollama's - it is hosted inference,
    not a CPU-bound local process."""
    model = chat_model(
        span=Span.ROUTE,
        settings=_settings(
            llm_provider="groq", groq_api_key="gsk-test", groq_model="llama-3.1-8b-instant"
        ),
    )
    assert model.model_name == "llama-3.1-8b-instant"
    assert model.request_timeout == 60


def test_client_side_retry_is_disabled() -> None:
    """The gateway owns retry policy. Retrying in both places multiplies attempts
    and hides the failure the gateway config is meant to surface."""
    assert chat_model(span=Span.ROUTE, settings=_settings()).max_retries == 0


def test_describe_deployment_reports_the_active_mode() -> None:
    assert "self-hosted" in describe_deployment(_settings())
    assert "hosted control plane" in describe_deployment(
        _settings(portkey_api_key="pk", portkey_config_app="pc")
    )


def test_describe_deployment_names_the_groq_model() -> None:
    described = describe_deployment(
        _settings(llm_provider="groq", groq_model="llama-3.1-8b-instant")
    )
    assert described.endswith("groq/llama-3.1-8b-instant")
