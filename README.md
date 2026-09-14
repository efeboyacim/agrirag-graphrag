# AgriRAG

Agentic **Graph RAG** over an agricultural knowledge graph.

Multi-hop agronomic questions ("which organic-approved treatments exist for maize
pests in Konya, and are any restricted there?") are answered by traversing a Neo4j
knowledge graph. Definitional questions ("what is integrated pest management?") are
answered by semantic search over LanceDB. A single LangGraph agent decides which
path to take, and every LLM call routes through a Portkey gateway.

> **Status: Phase 5 (evaluation).** All five phases are complete: the stack runs
> in Docker, answers over HTTP, and is measured by a DeepEval suite wired into CI.
> Evaluation lands next - see the roadmap.

## Stack

| Concern | Choice | Note |
|---|---|---|
| API | FastAPI, Pydantic v2, `/api/v1/*` | async throughout |
| Graph store | Neo4j 5.26 Community | behind a `GraphStore` Protocol, so it is swappable |
| Vector store | LanceDB | **embedded** - a directory, not a service |
| Agent | LangGraph | single stateful agent, explicit nodes |
| LLM gateway | Portkey | self-hosted OSS gateway by default |
| Evaluation | DeepEval | runs locally; Confident AI dashboard optional |

## Quick start

```bash
cp .env.example .env
# add ANTHROPIC_API_KEY to .env - needed from Phase 2 onward
docker compose up -d --build
uv run agrirag-seed --reset        # 75 nodes / 190 relationships
uv run agrirag-index --extract     # embed the corpus + link it into the graph
curl http://localhost:8000/health
```

Expected:

```json
{
  "status": "ok",
  "version": "0.1.0",
  "neo4j": "ok",
  "lancedb": "ok",
  "gateway": "ok",
  "chunks": 107,
  "llm_routing": "Portkey self-hosted OSS gateway at http://gateway:8787/v1"
}
```

Then ask it something:

```bash
curl -s -X POST http://localhost:8000/api/v1/ask   -H 'Content-Type: application/json'   -d '{"question":"Which organic-approved treatments exist for maize pests in Konya?"}'
```

The response carries the answer **and** the audit trail: the route taken and why,
the Cypher actually executed, the entities the question resolved to, and the full
retrieval context. You can check the answer was grounded rather than trust it.

Everything is reachable from Swagger at <http://localhost:8000/docs>.

- API docs: <http://localhost:8000/docs>
- Neo4j browser: <http://localhost:7474> (`neo4j` / `agrirag_dev_pw`)
- Portkey gateway UI: <http://localhost:8787/public/>

`agrirag-seed` needs no API key. `agrirag-index --extract` does - it runs LLM
extraction over the corpus. Plain `agrirag-index` (embeddings only) is free and
offline.

### Running without an API budget

The whole system runs against a local model - free, offline, no account:

```bash
ollama pull llama3.2
echo "LLM_PROVIDER=ollama" >> .env
docker compose up -d --build api
```

Cost in code: two config files plus provider-aware header assembly.

> **Run exactly one Ollama instance.** If `ollama serve` is started by hand while
> the desktop app is also running, the second loses the port and the model runner
> dies mid-request. It surfaces as an intermittent
> `ollama error: ... wsarecv: An existing connection was forcibly closed`, which
> looks like a bug in the agent - the symptom appears on whichever call happens
> to be in flight, not on the one that caused it. Check with
> `Get-Process *ollama*` and keep one. No node,
prompt, template or test changed - which is the concrete version of what the
gateway is for.

Retrieval is *identical* either way, because entity linking, Cypher templates and
vector search use no model at all. Only synthesis quality drops, and it drops a
lot: on "what should I rotate with wheat", Claude names all four partners with
per-pair rationale, llama3.2 names one. Use this to show the system works, not to
measure how well - and never read evaluation numbers from a local-model run.
See [docs/portkey.md](docs/portkey.md).

### Changing `.env` after the stack is running

Use `docker compose up -d`, not `restart`:

```bash
docker compose up -d api     # picks up .env changes (recreates the container)
docker compose restart api   # does NOT - env_file is read at creation time
```

`restart` stops and starts the same container, which keeps whatever environment
it was created with. This is measurably true, not folklore: a changed
`ANTHROPIC_API_KEY` survives a `restart` as the *old* value and the API keeps
failing authentication with no indication why.

## Local development

```bash
uv sync                          # provisions Python 3.12 and installs deps
uv run pytest tests/unit         # 132 unit tests, no services needed
uv run pytest tests/integration  # needs the stack; skips what is unreachable
uv run ruff check . && uv run mypy

uv run agrirag-seed --dry-run    # validate seed CSVs without a database
uv run agrirag-seed --reset      # wipe and reload the graph
uv run agrirag-index             # rebuild embeddings only (free, offline)
uv run agrirag-index --extract   # also run LLM extraction (costs API calls)

docker compose up -d --build
docker compose down              # volumes preserved
```

A `Makefile` wraps the same commands (`make check`, `make up`, `make help`) for CI and
for Linux/macOS/WSL. It is optional - `make` is not installed by default on Windows.

## Design notes

Three decisions worth calling out, because they are deliberate rather than accidental:

1. **LanceDB has no container.** It is an embedded, in-process store - a directory on a
   Docker volume. Reviewers expecting a vector-database service will not find one.
2. **The agent never imports a database driver.** It depends on the `GraphStore` and
   `VectorStore` Protocols. Replacing Neo4j with a managed graph means adding one
   implementation file, not editing agent code.
3. **All LLM traffic goes through one factory** (`agrirag.llm.portkey_client`), which is
   what makes "everything routes through the gateway" verifiable instead of aspirational.

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 0 | Scaffolding, Docker Compose, health endpoint, CI | **done** |
| 1 | Seed data, graph schema, ingestion, graph construction | **done** |
| 2 | Retrieval layer (Neo4j + LanceDB) and Portkey gateway | **done** |
| 3 | LangGraph agent: routing, retrieval, grading, synthesis | **done** |
| 4 | API layer: `/api/v1/ask`, `/ingest`, `/graph/schema` | **done** |
| 5 | Evaluation (DeepEval) and observability | **done** |

## The data

| | |
|---|---|
| Graph | 75 nodes, 190 relationships across 8 labels and 12 relationship types |
| Corpus | 28 agronomy documents, 107 chunks, 235 provenance links into the graph |
| Domain | Turkish agriculture - Konya, Cukurova, Aegean, Eastern Black Sea, Southeastern Anatolia |

Hand-authored rather than scraped, and that is a deliberate trade. Synthetic data
looks less impressive than a real scrape, but it means the Phase 5 evaluation
goldens have genuine ground truth: the expected entity ids for a question are
known because the same person wrote the question and the data.

See [docs/graph-schema.md](docs/graph-schema.md) for the full schema and the
modelling decisions behind it, [docs/agent.md](docs/agent.md) for the agent
design, and [docs/portkey.md](docs/portkey.md) for the gateway wiring.

## The agent

One LangGraph agent, not several. Six nodes:

```
START -> route
  route --conditional fan-out--> [graph_retrieve] | [semantic_retrieve] | both
  graph_retrieve / semantic_retrieve --> grade
  grade --> semantic_retrieve (broaden, once) | synthesize | abstain
```

Three things make this a real stateful agent rather than a RAG chain:

**Parallel fan-out.** The routing edge returns a *list* of node names, so on a
"both" question LangGraph runs the two retrievers in one superstep and merges
their writes through `operator.add` reducers on the state. Real concurrency, not
two sequential calls.

**A cycle.** `grade -> semantic_retrieve -> grade` retries a thin graph result
against the corpus with a wider `k`, capped at one pass.

**Abstention.** When the grader says the context cannot answer the question, the
agent declines instead of synthesising something plausible.

Routing and abstention measured over a 12-question spot check: **12/12**.

| Question | Route | Behaviour |
|---|---|---|
| Organic treatments for maize pests in Konya | `graph` | 4-hop traversal, 3 rows |
| Practices for wheat on clay that cut rust | `graph` | branching traversal, retried once |
| Explain integrated pest management | `semantic` | 5 chunks, graph untouched |
| Why cover cropping on sandy semi-arid soils | `both` | both paths, merged context |
| How do I configure a Kubernetes ingress? | - | **abstains** |

That last row is the one worth arguing about. Vector search always returns its k
nearest neighbours however irrelevant they are, so an early version answered
off-domain questions from agronomy chunks. A measured relevance floor fixed it -
see [docs/agent.md](docs/agent.md).

## Does the graph actually earn its place?

Worth testing rather than asserting. `force_route` skips the router, so you can
ask the same question with and without the graph:

```bash
curl -s -X POST http://localhost:8000/api/v1/ask -H 'Content-Type: application/json'   -d '{"question":"What should I rotate with wheat, and why?"}'

curl -s -X POST http://localhost:8000/api/v1/ask -H 'Content-Type: application/json'   -d '{"question":"What should I rotate with wheat, and why?","force_route":"semantic"}'
```

| | Graph | Semantic only |
|---|---|---|
| Rotation partners returned | **4** (chickpea, sunflower, sugar beet, cotton), each with its specific benefit | **1** (chickpea) |

`ROTATES_WITH` edges live only in the seed CSVs. The corpus discusses the
wheat-chickpea pairing at length and barely mentions the rest, so similarity
search finds the one document that exists and stops there. The graph returns the
complete set because completeness is a property of the traversal, not of whether
someone wrote a paragraph about it.

**The honest caveat:** this does not hold for every question. Ask
"which organic-approved treatments exist for maize pests in Konya?" both ways and
the answers are close, because the corpus happens to contain a fall-armyworm
document and a Konya document that between them cover it. Graph retrieval wins
where the answer is a *set of relationships no single document enumerates* - not
uniformly. Claiming otherwise would be easy to disprove in about thirty seconds,
which is exactly why the comparison is a built-in endpoint feature rather than a
paragraph of marketing.

## Retrieval

Two paths, deliberately different in kind.

**Graph** ([`agent/tools.py`](src/agrirag/agent/tools.py)) resolves entity
mentions to canonical node ids through a Neo4j fulltext index, selects a
parameterised Cypher template, validates it through a read-only guard, and
linearises the rows into sentences.

Templates rather than free-form text-to-Cypher: an LLM writing arbitrary Cypher
against an 8-label schema invents relationship directions and filters on
properties that do not exist, which returns zero rows and looks like a missing
fact rather than a lookup miss. Selecting from seven templates turns an
open-ended generation problem into a classification problem. Free-form generation
remains the fallback for questions no template covers.

Entity linking first is the other half of that reliability: the dominant
text-to-Cypher failure is filtering on `{name: 'corn'}` when the graph stores
`'Maize'`. Aliases carry the Turkish names, so `bugday` and `sune` resolve too.

**Semantic** ([`vector/`](src/agrirag/vector)) embeds the query locally with
`bge-small-en-v1.5` and searches LanceDB. No API key, no network, identical
vectors on every run - so re-indexing is free and the Phase 5 metrics are not
measuring embedding-API drift.

Both paths emit `context: list[str]`. That shared shape is what lets DeepEval's
contextual metrics score graph facts and text chunks with the same metric.

## API

| Endpoint | Purpose |
|---|---|
| `POST /api/v1/ask` | Ask a question. Returns answer, route, Cypher, retrieval context, citations |
| `POST /api/v1/ingest` | Rebuild the vector index; optionally re-run LLM extraction |
| `GET /api/v1/graph/schema` | Live schema with node and relationship counts |
| `GET /health` | Liveness plus per-dependency readiness |

Every response carries `x-request-id`, which is also the Portkey trace id - so one
API call's routing, grading and synthesis spans group under a single trace.

Infrastructure failures become actionable replies rather than stack traces: a
missing vector index returns 503 with the command to fix it, a missing API key
names the variable to set, and an unexpected error returns a generic message plus
the request id, keeping internals out of the response.

## Evaluation

35 hand-authored goldens, scored on two layers that read different things.
Full numbers in [evals/reports/latest.md](evals/reports/latest.md); method and
caveats in [docs/evaluation.md](docs/evaluation.md).

| Layer | Metric | Mean | Threshold |
|---|---|---:|---:|
| Retrieval | ContextualRecall | 0.92 | 0.80 |
| Retrieval | ContextualPrecision | 0.73 | 0.65 |
| Retrieval | ContextualRelevancy | 0.43 | 0.35 |
| Retrieval | **RoutingAccuracy** | **1.00** | 1.00 |
| Retrieval | **GraphPathRecall** | **1.00** | 1.00 |
| Generation | Faithfulness | 1.00 | 0.90 |
| Generation | AnswerRelevancy | 0.91 | 0.75 |
| Generation | Hallucination | 0.90 | 0.75 |
| Generation | AgronomicUsefulness | 0.93 | 0.75 |
| Generation | **CorrectAbstention** | **1.00** | 1.00 |

**Retrieval metrics read only `retrieval_context`; generation metrics read only
the answer.** That separation is the point - it makes a failure attributable to
fetching the wrong facts or to writing a bad answer over the right ones, which
need different fixes. It works because graph rows are linearised into sentences
before entering the context, so both retrieval paths arrive in the same shape.

**The three bold metrics are deterministic** - no model, no cost, no variance.
They gate every pull request without an API key, and when a judged score moves
they are how you tell a real regression from judge noise. Two consecutive report
runs over identical agent outputs moved judged metrics by up to 0.06 while these
three moved by 0.00, which is why judged thresholds sit ~0.10 below baseline and
are asserted on the mean rather than per case.

**ContextualRelevancy at 0.43 is the honest weak spot.** The first explanation
written for it - that graph traversal returns too many rows - was wrong, and the
data says so: graph rows score 0.53, corpus chunks 0.39. The metric grades each
statement inside each context item, so multi-sentence chunks lose points for the
sentences that do not bear on the question. The lever is chunk size, not
retrieval strategy.

The evaluation paid for itself during development by catching four bugs the
judged metrics only registered as vague low scores - a label missing from the
fulltext index, a non-stemming analyzer, a stopword regression, and the agent
answering a question it should have declined.

```bash
uv run python -m evals.runner --refresh   # run the agent over the goldens (~90s)
uv run python -m evals.report             # score once, write the report (~20m)
uv run pytest evals/                      # gate on it (~2s, no LLM calls)
```

## Documentation

| | |
|---|---|
| [docs/architecture.md](docs/architecture.md) | How the pieces fit and why each boundary sits where it does |
| [docs/graph-schema.md](docs/graph-schema.md) | Node labels, relationships, and the modelling decisions |
| [docs/agent.md](docs/agent.md) | Agent design: nodes, state, reducers, the grading cycle |
| [docs/portkey.md](docs/portkey.md) | Gateway wiring, fallback, and the gotchas that cost real time |
| [docs/evaluation.md](docs/evaluation.md) | Metrics, measured baselines, judge variance |

## Safety

Generated Cypher is read-only, enforced three independent ways:

1. [`cypher_guard.py`](src/agrirag/graph/cypher_guard.py) rejects write clauses
   and dangerous procedures, blocks statement stacking, and injects a `LIMIT`.
2. `Neo4jGraphStore` opens sessions in `READ_ACCESS` mode, so the server refuses
   a write that somehow got through.
3. The write path lives in `ingestion/writer.py`, which the agent never imports.

Any one would do. Three exist because part of the query text is LLM-generated,
and a prompt is not a security boundary.
