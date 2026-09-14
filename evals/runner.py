"""Run the agent over the golden set, once.

**Why results are cached to disk.** Scoring is split into two suites - retrieval
and generation - but both need the same agent outputs. Running the agent twice
would double the cost and, worse, score the two suites against *different* runs,
so a retrieval number and a generation number could never be compared.

Caching also separates "run the agent" from "score the outputs". Re-scoring after
a threshold change or a metric fix then costs nothing, which is what makes
iterating on the evaluation itself practical.

    uv run agrirag-eval-run            # run the agent, cache results
    uv run agrirag-eval-run --refresh  # ignore an existing cache
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agrirag.agent.graph import Agent, ask, build_agent
from agrirag.config import get_settings
from agrirag.graph.neo4j_store import Neo4jGraphStore
from agrirag.observability.logging import configure_logging
from agrirag.vector.lancedb_store import LanceDBVectorStore
from evals.dataset import Golden, active_goldens, describe

logger = logging.getLogger("agrirag.eval")

RUNS_DIR = Path(__file__).parent / "runs"
LATEST_RUN = RUNS_DIR / "latest.json"

#: Concurrent agent runs. Each golden is several LLM calls, so this is the knob
#: that decides whether a full evaluation takes two minutes or ten.
CONCURRENCY = 4


@dataclass
class AgentRun:
    """What the agent produced for one golden."""

    golden_id: str
    question: str
    answer: str
    route: str
    route_reason: str = ""
    retrieval_context: list[str] = field(default_factory=list)
    linked_entities: list[dict[str, Any]] = field(default_factory=list)
    abstained: bool = False
    template: str | None = None
    cypher: str | None = None
    graph_row_count: int = 0
    chunk_count: int = 0
    retries: int = 0
    duration_seconds: float = 0.0
    error: str | None = None


@asynccontextmanager
async def agent_session() -> AsyncIterator[Agent]:
    """Build an agent with no checkpointer.

    Stateless on purpose: a checkpointed evaluation would let one golden's state
    leak into the next, and thread reuse would quietly change results depending
    on execution order.
    """
    settings = get_settings()
    graph = Neo4jGraphStore(
        settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password, settings.neo4j_database
    )
    vectors = LanceDBVectorStore()
    try:
        await graph.verify_connectivity()
        yield build_agent(graph, vectors)
    finally:
        await graph.close()
        await vectors.close()


async def run_one(agent: Agent, golden: Golden) -> AgentRun:
    """Run a single golden, capturing failures rather than aborting the sweep."""
    started = time.perf_counter()
    try:
        result = await ask(agent, golden.input, trace_id=f"eval-{golden.id}")
    except Exception as exc:
        logger.error("golden %s failed: %s", golden.id, exc)
        return AgentRun(
            golden_id=golden.id,
            question=golden.input,
            answer="",
            route="error",
            error=f"{type(exc).__name__}: {exc}",
            duration_seconds=round(time.perf_counter() - started, 2),
        )

    return AgentRun(
        golden_id=golden.id,
        question=golden.input,
        answer=result.answer,
        route=result.route,
        route_reason=result.route_reason,
        retrieval_context=result.retrieval_context,
        linked_entities=result.linked_entities,
        abstained=result.abstained,
        template=result.template,
        cypher=result.cypher,
        graph_row_count=result.graph_row_count,
        chunk_count=result.chunk_count,
        retries=result.retries,
        duration_seconds=round(time.perf_counter() - started, 2),
    )


async def run_all(goldens: list[Golden]) -> dict[str, AgentRun]:
    """Run every golden with bounded concurrency."""
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async with agent_session() as agent:

        async def one(golden: Golden) -> AgentRun:
            async with semaphore:
                run = await run_one(agent, golden)
                logger.info(
                    "%-32s route=%-8s rows=%2d chunks=%2d abstained=%-5s %.1fs",
                    golden.id,
                    run.route,
                    run.graph_row_count,
                    run.chunk_count,
                    run.abstained,
                    run.duration_seconds,
                )
                return run

        runs = await asyncio.gather(*(one(golden) for golden in goldens))

    return {run.golden_id: run for run in runs}


def save(runs: dict[str, AgentRun], path: Path = LATEST_RUN) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runs": {gid: asdict(run) for gid, run in runs.items()},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("wrote %d agent runs to %s", len(runs), path)


def load(path: Path = LATEST_RUN) -> dict[str, AgentRun]:
    """Load cached runs. Raises if the cache is absent."""
    if not path.exists():
        raise FileNotFoundError(f"no cached agent runs at {path}. Run: uv run agrirag-eval-run")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {gid: AgentRun(**data) for gid, data in payload["runs"].items()}


def load_or_fail(goldens: list[Golden], path: Path = LATEST_RUN) -> dict[str, AgentRun]:
    """Load cached runs and check they cover the current golden set.

    A stale cache is worse than a missing one: it scores today's thresholds
    against yesterday's questions and reports a clean pass.
    """
    runs = load(path)
    missing = [g.id for g in goldens if g.id not in runs]
    if missing:
        raise RuntimeError(
            f"cached run is stale - missing {len(missing)} golden(s): {missing[:5]}. "
            "Re-run: uv run agrirag-eval-run --refresh"
        )
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the agent over the golden set.")
    parser.add_argument("--refresh", action="store_true", help="ignore any existing cache")
    parser.add_argument("--only", help="run a single golden by id")
    args = parser.parse_args()

    configure_logging(get_settings().log_level)

    goldens = active_goldens()
    if args.only:
        goldens = [g for g in goldens if g.id == args.only]
        if not goldens:
            logger.error("no golden with id %r", args.only)
            return 1

    dist = describe(goldens)
    logger.info("golden set: %d cases", dist.total)
    logger.info("  by category : %s", dist.by_category)
    logger.info("  by route    : %s", dist.by_route)
    logger.info("  abstain     : %d", dist.abstain_cases)
    logger.info("  multi-hop   : %d", dist.multi_hop_cases)

    if LATEST_RUN.exists() and not args.refresh and not args.only:
        try:
            load_or_fail(goldens)
            logger.info("cache at %s is current; pass --refresh to re-run", LATEST_RUN)
            return 0
        except RuntimeError as exc:
            logger.info("%s", exc)

    started = time.perf_counter()
    runs = asyncio.run(run_all(goldens))
    save(runs)

    failed = [r for r in runs.values() if r.error]
    logger.info("completed %d runs in %.0fs", len(runs), time.perf_counter() - started)
    if failed:
        logger.error("%d runs errored: %s", len(failed), [r.golden_id for r in failed])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
