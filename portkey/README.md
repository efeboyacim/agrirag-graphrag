# Portkey configuration

Two configs so that application traffic and evaluation traffic are separable in
the traces - otherwise a nightly eval run makes the request volume and cost of the
API itself impossible to read.

| File | Used by | Cache TTL |
|---|---|---|
| `config.app.json` | router, Cypher generation, extraction, grading, synthesis | 1 hour |
| `config.eval.json` | the DeepEval judge model | 24 hours |

Each has an `.ollama.json` and a `.groq.json` variant; `LLM_PROVIDER` picks
which file is loaded, and nothing in Python changes.

## Fallback chain

The chain is model-level within the selected provider (`LLM_PROVIDER`):

```
anthropic : claude-sonnet-5          ->  claude-haiku-4-5-20251001
groq      : llama-3.3-70b-versatile  ->  llama-3.1-8b-instant
ollama    : llama3.2 (single target)
```

Haiku is the fallback because it survives rate limits and overload conditions
that throttle the larger model. This exercises the same gateway machinery a
multi-provider chain would - `strategy.mode`, `on_status_codes`, per-target
retry, per-target traces - and adding a second provider later is an edit to this
file, not to any Python.

The eval config deliberately has no fallback: a judge that silently degrades to a
smaller model would make metric scores incomparable between runs.

## Caching is declared but inactive here

Both configs declare a `cache` block. On the self-hosted OSS gateway it has no
effect - the gateway returns `x-portkey-cache-status: DISABLED` even for two
identical back-to-back requests. Caching requires the hosted control plane's
cache store.

The declaration is kept because it is correct and activates as soon as
`PORTKEY_API_KEY` is set. It is documented here so nobody reads the TTL column
below and concludes that repeated runs are being served from cache.

## Deployment

Default is the self-hosted OSS gateway (`portkeyai/gateway`, the `gateway`
compose service). It needs no account, which keeps `docker compose up` working
with nothing but an Anthropic key.

With the OSS gateway these files are sent inline on every request in the
`x-portkey-config` header. On the hosted control plane the same JSON would be
saved as a config and referenced by id via `PORTKEY_CONFIG_APP` /
`PORTKEY_CONFIG_EVAL`; the client reads those env vars and switches
automatically, so moving to the hosted plane is configuration only.
