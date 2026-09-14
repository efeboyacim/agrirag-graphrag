# Portkey Integration

Every LLM call in AgriRAG goes through a Portkey gateway. This document covers how
that is wired, why the choices were made, and the three gateway-specific gotchas
that cost real debugging time.

## The chokepoint

[`llm/portkey_client.py`](../src/agrirag/llm/portkey_client.py) is the only module
that constructs an LLM client. The router, Cypher generation, entity extraction,
grading, synthesis and the Phase 5 evaluation judge all call `chat_model()` or
`structured_model()`.

That claim is enforced, not documented:
`tests/unit/test_portkey_client.py::test_no_module_constructs_an_llm_outside_the_factory`
scans the source tree and fails if anything outside `llm/` imports
`langchain_openai`, `langchain_anthropic`, or the `anthropic` SDK. Without it,
someone reaching for `ChatAnthropic` to "just quickly test something" would bypass
fallback, retry and tracing, and nothing would notice.

## Why ChatOpenAI against an Anthropic provider

The gateway exposes an OpenAI-compatible surface and translates to the Anthropic
API itself. So the client is `ChatOpenAI` pointed at `PORTKEY_BASE_URL`, with
`x-portkey-provider: anthropic`.

Using `ChatAnthropic` would talk to `api.anthropic.com` directly and silently lose
everything the gateway provides. That is the failure this module exists to prevent.

## Deployment modes

| | Self-hosted OSS gateway (default) | Hosted control plane |
|---|---|---|
| Selected by | `PORTKEY_API_KEY` empty | `PORTKEY_API_KEY` set |
| Config delivery | full JSON inline in `x-portkey-config` | config id in `x-portkey-config` |
| Credentials | injected into each target at request time | virtual keys |
| Account needed | no | yes (free tier) |
| Observability | request logs + a UI at `:8787/public/` | full dashboard, traces, cost |

Switching is configuration, not code. `describe_deployment()` reports the active
mode, and `/health` surfaces it in `llm_routing`.

**What the OSS gateway does and does not do.** Measured, not assumed:

| Capability | OSS gateway | Hosted plane |
|---|---|---|
| Provider abstraction | yes | yes |
| Fallback chain | yes - verified by test | yes |
| Retry / `on_status_codes` | yes | yes |
| **Response caching** | **no** | yes |
| Traces, cost, latency per span | request logs only | dashboard |

The cache result is worth stating plainly because the committed configs declare
`"cache": {"mode": "simple", "max_age": 3600}` and it has no effect: the gateway
returns `x-portkey-cache-status: DISABLED` on every request, on both a first and
an identical repeat call. Caching needs the hosted control plane's cache store.
The config block is kept because it is the correct declaration and starts working
the moment `PORTKEY_API_KEY` is set - but nothing in this repo is currently
served from cache, and any claim that repeated evaluation runs are cheaper
because of it would be false.

**Recommendation for the demo:** the OSS gateway proves fallback, retry and
provider abstraction with no account. The hosted plane adds caching and makes
traces, cost and per-target latency *visible*, which is most of what this
component is worth showing. Free signup, one env var, no code change.

## Running without an API budget: the local provider

Set `LLM_PROVIDER=ollama` and the whole system runs against a local model - free,
offline, no account:

```bash
ollama pull llama3.2
echo "LLM_PROVIDER=ollama" >> .env
docker compose up -d --build api     # up -d, not restart - see the README
```

**What this cost in code: two config files and provider-aware header assembly.**
`load_config()` picks `config.app.ollama.json` instead of `config.app.json`; no
node, prompt, template or test changed. That is the concrete version of the claim
this whole component exists to make - swapping a provider is configuration.

### What actually changed in behaviour

Measured against llama3.2 (3B) on the same questions:

| | Anthropic (sonnet-5) | Ollama (llama3.2 3B) |
|---|---|---|
| Graph + semantic retrieval | identical | identical - no model involved |
| Routing | correct | mostly correct, after tolerance fixes |
| Abstention on off-domain | correct | correct |
| Answer depth | names all four rotation partners with per-pair rationale | names one, briefly |
| Speed | ~10 s | ~60-120 s on CPU |

Retrieval is unaffected because it does not use a model at all - entity linking,
Cypher templates and vector search are deterministic. Only synthesis quality
drops, and it drops a lot.

### One change the local model forced, which was worth making anyway

`RouteDecision` now coerces loose values: `"Graph"`, `" semantic "`, a sentence
containing the word, and `"no"` for a boolean. llama3.2 produced all of these and
strict Pydantic rejected every one, so *every* routing call failed and the agent
fell back to running both retrievers on every question.

The validators only fire when the raw value is not already valid, so a model that
fills the schema correctly is unaffected. This is a robustness improvement for any
weaker or cheaper model, not an Ollama workaround.

### What had to change for the agent to complete at all

A question normally costs three LLM calls - route, grade, synthesize. On a
CPU-bound 3B model each takes 30-90 seconds and the Ollama runner becomes
unstable under that load: roughly half of all requests died mid-flight with
`wsarecv: An existing connection was forcibly closed`, and the error surfaced on
whichever call happened to be in flight rather than the one that caused it.

Four provider-aware defaults fixed it, each with an explicit override:

| Setting | Anthropic | Ollama | Why |
|---|---|---|---|
| `LLM_TIMEOUT_SECONDS` | 60 | 300 | a local model answers in minutes, not seconds |
| `MAX_CONTEXT_CHARS` | unlimited | 2200 | the runner dies above roughly 2000 tokens of input |
| router prompt | full (~1670 tokens) | compact (~235) | the full prompt alone sat at the crash ceiling |
| `LLM_GRADING` | on | off | three calls per question is one too many here |

Turning grading off does **not** disable abstention. The grader's empty-context
branch is deterministic and still runs, so off-domain questions are declined as
before - verified. What is lost is the narrower judgement "context exists, is on
topic, but does not answer *this* question".

Result: every request completes, routing and abstention are correct, and a
question takes ~30 seconds instead of ~3.5 minutes.

### What it is not suitable for

**Answer quality is the limit, and it is a hard one.** Asked what to rotate with
wheat - with the correct four partners sitting in the retrieval context -
llama3.2 answered "rotate wheat with barley", then "barley or oats", then
"rotate wheat with sunn pest" (an insect, not a crop). Retrieval was right every
time; the model did not use it. Claude scores 1.00 on faithfulness over the same
goldens.

So the local provider demonstrates that the *system* works. It does not produce
answers anyone should show to a domain expert.

**Do not read evaluation numbers from a local-model run.** The committed baseline
in `evals/reports/latest.md` was produced with Claude as both the system model and
the judge. A 3B judge grading a 3B system produces numbers that are not comparable
to it and not meaningful on their own. Use the local provider to demonstrate that
the system *works*, not to measure how well.

## Configuration

Two configs, committed in [`portkey/`](../portkey):

- `config.app.json` - router, Cypher generation, extraction, grading, synthesis.
  Fallback chain, 1-hour cache declared (inactive on the OSS gateway - see above).
- `config.eval.json` - the DeepEval judge. Single target, 24-hour cache declared.

Separating them means a nightly eval run does not make the API's own request
volume and cost unreadable. The eval config deliberately has **no fallback**: a
judge that silently degraded to a smaller model would make metric scores
incomparable between runs.

### Fallback chain

Anthropic is the only provider, so the chain is model-level:

```
claude-sonnet-5  ->  claude-haiku-4-5-20251001
```

Haiku survives the rate limits and overload conditions that throttle the larger
model. This exercises exactly the same machinery a multi-provider chain would -
`strategy.mode`, `on_status_codes`, per-target retry, per-target traces - and
adding a second provider is an edit to a JSON file, not to any Python.

`on_status_codes` is `[429, 500, 502, 503, 529]`. It deliberately excludes 404: a
missing model is a configuration mistake, not the transient failure fallback
exists for, and masking it would hide a real bug.

### Credentials

The committed config files contain no secrets - a unit test asserts that. The
provider key is injected into each target at request time by `_resolve_config()`,
on a deep copy, so the file on disk stays clean. On the hosted plane a virtual key
does this job instead.

## What gets traced

Every request carries:

| Field | Value |
|---|---|
| `trace_id` | the FastAPI request id - groups every LLM span from one API call |
| `span` | `route` / `cypher_gen` / `extract` / `grade` / `synthesize` / `eval_judge` |
| `purpose` | `app` or `eval` |
| extras | `chunk_id`, `doc_id` during ingestion; the golden id during evaluation |

The `span` field is what turns a trace from an undifferentiated pile of
completions into something readable during a demo.

## Three gotchas worth knowing

These were all found by testing, and each fails in a way that is easy to miss.

**1. `temperature` is rejected.** The Claude 5 models return
`400 temperature is deprecated for this model`. `chat_model()` therefore defaults
`temperature` to `None` and omits it from the payload entirely. These calls are
classification and structured extraction, where sampling control would not help
anyway.

**2. Only `function_calling` structured output survives the translation.**

| Method | Result |
|---|---|
| `function_calling` | works - translated to Anthropic's native tool use (`toolu_` ids) |
| `json_schema` | fails - the translated response has no `parsed` field |
| `json_mode` | fails - replies with fenced markdown, sometimes YAML |

`structured_model()` hard-codes `method="function_calling"` so no caller can
reintroduce the mistake. The failure mode is nasty: extraction returns nothing for
every chunk while the pipeline reports success.

**3. Nested tool arguments are sometimes stringified.** Anthropic-via-Portkey
intermittently returns a nested array as a JSON *string* rather than an array -
and not consistently, so the same prompt succeeds on one call and stringifies on
the next. `ExtractionResult` has a `mode="before"` validator that parses it. Worth
knowing before discovering it midway through a 107-chunk batch.

A fourth, not gateway-specific but adjacent: `langchain_openai` pools httpx
clients across `ChatOpenAI` instances. Under pytest-asyncio's default per-test
event loop, the second live test reuses a connection bound to a closed loop and
fails with an opaque `Connection error`. The suite runs on a session-scoped loop
(`asyncio_default_test_loop_scope = "session"`) for this reason.

## Verifying it works

```bash
uv run pytest tests/integration/test_gateway.py -v
```

Three live tests (they skip without `ANTHROPIC_API_KEY`, which is how CI runs
them):

1. A request reaches Anthropic through the gateway and reports the primary model.
2. **Fallback engages** - the primary is pointed at a nonexistent model and the
   reply comes back from `claude-haiku-4-5`. Asserted over raw HTTP, because
   fallback is a gateway property and putting a LangChain client in the middle
   only adds the connection-pool failure mode described above.
3. Structured output survives the translation layer.

To watch it by hand:

```bash
docker compose logs -f gateway     # request log
open http://localhost:8787/public/ # the OSS gateway UI
```
