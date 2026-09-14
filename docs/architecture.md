# Architecture

How the pieces fit, and why each boundary is where it is. Component detail lives
in [agent.md](agent.md), [graph-schema.md](graph-schema.md),
[portkey.md](portkey.md) and [evaluation.md](evaluation.md).

## The whole system

```
                    ┌───────────────────────────────────────┐
                    │  Client: Swagger UI / curl / eval run  │
                    └────────────────────┬──────────────────┘
                                         │ POST /api/v1/ask
                    ┌────────────────────▼──────────────────┐
                    │  FastAPI  (container: api)             │
                    │  /ask  /ingest  /graph/schema  /health │
                    │  request-id middleware -> trace id     │
                    └────────────────────┬──────────────────┘
                                         │ await agent.ainvoke()
                    ┌────────────────────▼──────────────────┐
                    │  LangGraph agent  (in-process)         │
                    │  route → [graph ‖ semantic] → grade    │
                    │        → synthesize | abstain          │
                    │  checkpointer: SQLite (thread_id)      │
                    └──┬──────────────┬──────────────┬──────┘
                       │              │              │
              GraphStore        VectorStore     every LLM call
              (Protocol)        (Protocol)            │
                       │              │              │
              ┌────────▼─────┐ ┌──────▼──────┐ ┌─────▼──────────────┐
              │   Neo4j 5.26 │ │  LanceDB    │ │  Portkey gateway   │
              │  (container) │ │ (EMBEDDED - │ │  (container, OSS)  │
              │  bolt :7687  │ │  a volume,  │ │  :8787             │
              │  READ-mode   │ │  no server) │ │  fallback / retry  │
              └──────────────┘ └─────────────┘ └─────┬──────────────┘
                                                     │
                                       ┌─────────────┴──────────────┐
                                       │ Anthropic                  │
                                       │ sonnet-5 → haiku-4.5       │
                                       └────────────────────────────┘

   Ingestion (offline)
   data/seed/*.csv ──► graph_builder ──► Neo4j backbone   (deterministic)
   data/corpus/*.md ─► chunker ─┬─► extractor (LLM) ─► (:Chunk)-[:MENTIONS]->
                                └─► fastembed (local) ─► LanceDB

   Evaluation (offline)
   evals/goldens/*.yaml ─► runner ─► cached AgentResults ─► report ─► pytest gate
```

## Four boundaries that carry the design

### 1. The agent depends on Protocols, never on drivers

`agent/` imports `GraphStore` and `VectorStore` — Protocol classes with no
implementation. It never imports `neo4j` or `lancedb`.

Swapping Neo4j for a managed graph means writing one new implementation file and
changing one line of wiring in the app lifespan. Agent logic, prompts, templates
and tests are untouched. That is the swappability requirement satisfied with a
Protocol rather than an ORM or a framework.

### 2. One LLM chokepoint

`llm/portkey_client.py` is the only module that constructs an LLM client. Router,
Cypher generation, extraction, grading, synthesis and the evaluation judge all go
through it.

This is enforced rather than documented: a unit test scans the source tree and
fails if anything outside `llm/` imports `langchain_openai`, `langchain_anthropic`
or the `anthropic` SDK. Without that test, someone reaching for `ChatAnthropic` to
"just quickly test something" would bypass fallback, retry and tracing, and
nothing would notice.

Stopping the gateway container proves the property: the API returns
`router unavailable (OpenAIConnectionError)` and abstains. There is no second path
to a provider.

### 3. Read and write are separate paths

| | Query time | Ingestion |
|---|---|---|
| Class | `Neo4jGraphStore` | `Neo4jWriter` |
| Session mode | `READ_ACCESS` | `WRITE_ACCESS` |
| Imported by | the agent | `ingestion/` and the `/ingest` route only |

Generated Cypher cannot mutate the graph, enforced three independent ways: the
`cypher_guard` rejects write clauses and injects a `LIMIT`, the server refuses a
write on a READ session, and the agent never imports the writer at all. Any one
would do; three exist because part of the query text is LLM-generated and a prompt
is not a security boundary.

### 4. Retrieval outputs share one shape

Both retrieval paths emit `context: list[str]`. Graph rows are linearised into
sentences through their template's format string before they get there.

This is what makes the evaluation split possible. DeepEval's contextual metrics
can then score graph facts and prose chunks with the same metric — otherwise one
is a table and the other is text, and there is no common ground to measure.

## What runs where

| | Where | Why |
|---|---|---|
| Neo4j | container | needs a server |
| Portkey gateway | container | needs a server |
| **LanceDB** | **in-process** | embedded; a directory on a volume, not a service |
| **Embeddings** | **in-process** | ONNX via fastembed: no key, no network, deterministic |
| Agent | in-process | compiled once at startup |
| Checkpoints | SQLite file | conversations survive a restart |

Two of those surprise people. There is no vector-database container — LanceDB is a
library, and the store is `./data/lancedb` bind-mounted into the API. And
embeddings never leave the machine, which is why re-indexing the corpus is free
and the evaluation is not measuring embedding-API drift.

## Request lifecycle

1. Middleware assigns a request id. It becomes the Portkey `trace_id`, so every
   LLM span raised while serving one call groups under one trace.
2. `route` classifies the question and names a Cypher template — one structured
   LLM call.
3. Retrieval fans out. On `both`, LangGraph runs the two retrievers in the same
   superstep and merges their writes through `operator.add` reducers.
4. `grade` decides whether the context can answer the question.
5. `synthesize` writes a grounded answer, or `abstain` declines.
6. The response carries the answer plus the audit trail: route and reason, the
   Cypher actually executed, resolved entities, and the full retrieval context.

## Data flow, and what is deterministic

The graph has two layers, and the distinction matters for evaluation stability:

- **Backbone** — 75 nodes and 190 relationships loaded from committed CSVs.
  Byte-identical on every run.
- **Provenance** — `(:Chunk)-[:MENTIONS]->(:Entity)` written by an LLM extraction
  pass over the corpus. Additive only: extraction *links* to entities that
  already exist and can never create one, so a mention the model invents is
  counted and dropped.

If extraction could create entities, the graph would differ between runs and every
evaluation threshold would drift with it. Keeping the backbone deterministic is
what lets the goldens have real ground truth.

## Configuration

One `Settings` object (`config.py`, pydantic-settings). Nothing reads `os.environ`
directly. `.env.example` documents every variable; only `ANTHROPIC_API_KEY` has no
working default.

Container-internal values (`bolt://neo4j:7687`, `http://gateway:8787/v1`) are set
in compose and override `.env`, so the same file works for host-run CLI tools and
for the containers.
