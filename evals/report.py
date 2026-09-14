"""Score every metric over the golden set and write the report.

This is the measurement tool; the pytest suites are the CI gate. They overlap
deliberately: a report is what you read, a threshold is what fails a build, and
conflating them tends to produce thresholds nobody can justify.

    uv run python -m evals.report

Writes evals/reports/latest.json (raw per-case scores, gitignored) and
evals/reports/latest.md (the committed summary).
"""

import argparse
import asyncio
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from deepeval.metrics import (
    AnswerRelevancyMetric,
    BaseMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    GEval,
    HallucinationMetric,
)
from deepeval.test_case import SingleTurnParams

from agrirag.config import get_settings
from agrirag.observability.logging import configure_logging
from evals import thresholds
from evals.cases import build_case
from evals.dataset import Golden, active_goldens, describe
from evals.judge import get_judge
from evals.metrics import CorrectAbstention, GraphPathRecall, RoutingAccuracy
from evals.runner import AgentRun, load_or_fail

logger = logging.getLogger("agrirag.eval.report")

REPORTS_DIR = Path(__file__).parent / "reports"

#: Cases scored concurrently. Metrics *within* a case run sequentially.
#:
#: Deliberately low. Scoring 5 cases at once with all 7 metrics in parallel put
#: 35 large judge calls in flight and tripped the Anthropic workspace limit of
#: 100k input tokens per minute. Every rate-limited metric then scores 0.0, so
#: the report read as a catastrophic quality collapse when nothing was wrong
#: with the system at all - which is a good argument for making metric errors
#: visibly distinct from genuine low scores.
CONCURRENCY = 2

#: Refuse to publish a report when more than this fraction of metric results
#: failed to compute. Errors score 0.0, so a run with widespread failures looks
#: identical to a total quality collapse - and silently replaces a good baseline.
MAX_ERROR_RATE = 0.10

#: Metrics where a *lower* score is better. Empty - DeepEval 4.x scores
#: HallucinationMetric as alignment, so every metric here is higher-is-better.
#: Kept as a named concept because it is exactly the kind of assumption that is
#: easy to get backwards and hard to notice.
LOWER_IS_BETTER: set[str] = set()


@dataclass
class MetricResult:
    name: str
    layer: str
    score: float
    passed: bool
    threshold: float
    reason: str = ""
    #: True when the metric could not be computed at all.
    #:
    #: Tracked separately from a low score because they are different events and
    #: were repeatedly confused during development: a rate limit, a malformed
    #: judge response and an invalidated API key each produced a 0.0 that looked
    #: exactly like catastrophically bad retrieval. An errored metric is a
    #: *missing measurement*, not a bad one.
    errored: bool = False


@dataclass
class CaseResult:
    golden_id: str
    category: str
    route: str
    abstained: bool
    metrics: list[MetricResult] = field(default_factory=list)


def _usefulness(judge: Any) -> GEval:
    return GEval(
        name="AgronomicUsefulness",
        criteria=(
            "Judge whether the answer would let a farmer or agronomist act. A good "
            "answer names the specific crops, pests, inputs or practices involved and "
            "carries across concrete values present in the context - application rates, "
            "pre-harvest intervals, efficacy, pH ranges. It states plainly when part of "
            "the question cannot be answered from the context rather than padding. A "
            "poor answer is vague, omits concrete values that were available, or hedges "
            "without saying what is missing."
        ),
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.RETRIEVAL_CONTEXT,
        ],
        model=judge,
        threshold=thresholds.AGRONOMIC_USEFULNESS,
        async_mode=False,
    )


async def _score(
    metric: BaseMetric, case: Any, name: str, layer: str, threshold: float
) -> MetricResult:
    try:
        await metric.a_measure(case)
        score = float(metric.score or 0.0)
        passed = score <= threshold if name in LOWER_IS_BETTER else bool(metric.is_successful())
        return MetricResult(name, layer, round(score, 3), passed, threshold, metric.reason or "")
    except Exception as exc:
        logger.warning("%s failed on %s: %s", name, case.name, exc)
        return MetricResult(
            name, layer, 0.0, False, threshold, f"metric error: {exc}", errored=True
        )


async def score_case(golden: Golden, run: AgentRun, judge: Any) -> CaseResult:
    """Score one golden with every applicable metric."""
    case = build_case(golden, run)
    result = CaseResult(golden.id, golden.category, run.route, run.abstained)

    # Deterministic first: free, and they explain the judged results.
    for metric, name in (
        (RoutingAccuracy(golden), "RoutingAccuracy"),
        (GraphPathRecall(golden), "GraphPathRecall"),
        (CorrectAbstention(golden), "CorrectAbstention"),
    ):
        metric.measure(case)
        layer = "generation" if name == "CorrectAbstention" else "retrieval"
        result.metrics.append(
            MetricResult(
                name,
                layer,
                round(float(metric.score), 3),
                metric.is_successful(),
                1.0,
                metric.reason,
            )
        )

    if golden.must_abstain:
        # No retrieval context and no answer to score. Declining correctly is
        # measured by CorrectAbstention above; running contextual metrics over
        # an empty context produces noise, not a zero.
        return result

    judged: list[tuple[BaseMetric, str, str, float]] = [
        (
            ContextualRecallMetric(
                threshold=thresholds.CONTEXTUAL_RECALL, model=judge, async_mode=False
            ),
            "ContextualRecall",
            "retrieval",
            thresholds.CONTEXTUAL_RECALL,
        ),
        (
            ContextualPrecisionMetric(
                threshold=thresholds.CONTEXTUAL_PRECISION, model=judge, async_mode=False
            ),
            "ContextualPrecision",
            "retrieval",
            thresholds.CONTEXTUAL_PRECISION,
        ),
        (
            ContextualRelevancyMetric(
                threshold=thresholds.CONTEXTUAL_RELEVANCY, model=judge, async_mode=False
            ),
            "ContextualRelevancy",
            "retrieval",
            thresholds.CONTEXTUAL_RELEVANCY,
        ),
        (
            FaithfulnessMetric(threshold=thresholds.FAITHFULNESS, model=judge, async_mode=False),
            "Faithfulness",
            "generation",
            thresholds.FAITHFULNESS,
        ),
        (
            AnswerRelevancyMetric(
                threshold=thresholds.ANSWER_RELEVANCY, model=judge, async_mode=False
            ),
            "AnswerRelevancy",
            "generation",
            thresholds.ANSWER_RELEVANCY,
        ),
        (_usefulness(judge), "AgronomicUsefulness", "generation", thresholds.AGRONOMIC_USEFULNESS),
    ]
    if golden.context:
        judged.append(
            (
                HallucinationMetric(
                    threshold=thresholds.HALLUCINATION, model=judge, async_mode=False
                ),
                "Hallucination",
                "generation",
                thresholds.HALLUCINATION,
            )
        )

    # Sequential, not gathered: gathering here multiplies in-flight calls by the
    # number of metrics and is what tripped the rate limit.
    for metric, name, layer, threshold in judged:
        result.metrics.append(await _score(metric, case, name, layer, threshold))
    return result


async def score_all(goldens: list[Golden], runs: dict[str, AgentRun]) -> list[CaseResult]:
    judge = get_judge("eval-report")
    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def one(golden: Golden) -> CaseResult:
        async with semaphore:
            result = await score_case(golden, runs[golden.id], judge)
            failed = [m.name for m in result.metrics if not m.passed]
            logger.info("%-32s %s", golden.id, "OK" if not failed else f"below threshold: {failed}")
            return result

    return list(await asyncio.gather(*(one(g) for g in goldens)))


def aggregate(results: list[CaseResult]) -> dict[str, dict[str, Any]]:
    """Mean, median and pass rate per metric."""
    buckets: dict[str, list[MetricResult]] = {}
    for case in results:
        for metric in case.metrics:
            buckets.setdefault(metric.name, []).append(metric)

    summary: dict[str, dict[str, Any]] = {}
    for name, metrics in buckets.items():
        scores = [m.score for m in metrics]
        summary[name] = {
            "layer": metrics[0].layer,
            "threshold": metrics[0].threshold,
            "lower_is_better": name in LOWER_IS_BETTER,
            "n": len(scores),
            "mean": round(statistics.mean(scores), 3),
            "median": round(statistics.median(scores), 3),
            "min": round(min(scores), 3),
            "max": round(max(scores), 3),
            "pass_rate": round(sum(m.passed for m in metrics) / len(metrics), 3),
            "failures": [m.name for m in metrics if not m.passed],
        }
    return summary


def _table(summary: dict[str, dict[str, Any]], layer: str) -> str:
    rows = [
        "| Metric | n | Mean | Median | Min | Threshold | Pass rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stats in sorted(summary.items()):
        if stats["layer"] != layer:
            continue
        direction = "<=" if stats["lower_is_better"] else ">="
        rows.append(
            f"| {name} | {stats['n']} | {stats['mean']:.2f} | {stats['median']:.2f} | "
            f"{stats['min']:.2f} | {direction} {stats['threshold']:.2f} | "
            f"{stats['pass_rate'] * 100:.0f}% |"
        )
    return "\n".join(rows)


def render_markdown(
    results: list[CaseResult], summary: dict[str, dict[str, Any]], goldens: list[Golden]
) -> str:
    dist = describe(goldens)
    failing = [
        (c.golden_id, m.name, m.score, m.reason) for c in results for m in c.metrics if not m.passed
    ]

    lines = [
        "# Evaluation Report",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} against {dist.total} goldens.",
        "",
        "Metrics are split by what they read. Retrieval metrics look only at",
        "`retrieval_context`; generation metrics look only at the answer. That split is",
        "the point: it makes a failure attributable to fetching the wrong facts or to",
        "writing a bad answer over the right ones, which need different fixes.",
        "",
        "## Golden set",
        "",
        "| | |",
        "|---|---|",
        f"| Total | {dist.total} |",
        f"| By category | {', '.join(f'{k}: {v}' for k, v in sorted(dist.by_category.items()))} |",
        f"| Multi-hop (2+) | {dist.multi_hop_cases} |",
        f"| Must abstain | {dist.abstain_cases} |",
        "",
        "## Retrieval quality",
        "",
        _table(summary, "retrieval"),
        "",
        "## Generation quality",
        "",
        _table(summary, "generation"),
        "",
        "## Reading these numbers",
        "",
        "- **RoutingAccuracy, GraphPathRecall, CorrectAbstention** are deterministic and",
        "  free. They gate every CI run, and when a judged score moves they are how you",
        "  tell a real regression from judge variance.",
        "- **ContextualRelevancy scores lowest by design.** A graph traversal returns",
        "  every row matching the pattern, and rows that are correct but not needed for",
        "  this particular question count against it. That is a real cost of graph",
        "  retrieval; the threshold reflects it rather than hiding it.",
        "- **Hallucination is scored against hand-written ground truth**, not against",
        "  what retrieval returned. An answer can be perfectly faithful to bad context",
        "  and still be false, and only this metric separates the two. Lower is better.",
        "- **Abstention cases carry no contextual scores.** They have no retrieval",
        "  context by design, so the contextual metrics have nothing to measure.",
        "",
    ]

    if failing:
        lines += [
            "## Below threshold",
            "",
            "| Golden | Metric | Score | Reason |",
            "|---|---|---:|---|",
        ]
        for gid, name, score, reason in sorted(failing):
            lines.append(f"| `{gid}` | {name} | {score:.2f} | {reason[:160].replace('|', '/')} |")
    else:
        lines += ["## Below threshold", "", "None - every metric met its threshold."]

    lines += [
        "",
        "## Reproducing",
        "",
        "```bash",
        "docker compose up -d",
        "uv run agrirag-seed --reset && uv run agrirag-index --extract",
        "uv run python -m evals.runner --refresh   # run the agent over the goldens",
        "uv run python -m evals.report             # score and regenerate this file",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Score the golden set and write the report.")
    parser.add_argument("--fail-under", action="store_true", help="exit non-zero on any failure")
    args = parser.parse_args()

    configure_logging(get_settings().log_level)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    goldens = active_goldens()
    runs = load_or_fail(goldens)

    started = time.perf_counter()
    results = asyncio.run(score_all(goldens, runs))
    summary = aggregate(results)

    total = sum(len(c.metrics) for c in results)
    errored = sum(1 for c in results for m in c.metrics if m.errored)
    error_rate = errored / total if total else 0.0

    if error_rate > MAX_ERROR_RATE:
        # Refuse to write. A run this broken is not a measurement, and
        # overwriting a valid report with it destroys the baseline while looking
        # like a catastrophic quality regression.
        #
        # This guard exists because it happened: an API key was invalidated
        # midway through a run, 338 calls returned 401, every affected metric
        # scored 0.0, and the committed report went from a 0.92 mean to 0.35.
        # Nothing was wrong with the system.
        logger.error(
            "%d of %d metric results errored (%.0f%%) - refusing to overwrite %s. "
            "Fix the cause and re-run; the previous report is untouched.",
            errored,
            total,
            error_rate * 100,
            REPORTS_DIR / "latest.md",
        )
        sample = next(
            (m.reason for c in results for m in c.metrics if m.errored), "no reason captured"
        )
        logger.error("first error was: %s", sample[:300])
        return 1

    if errored:
        logger.warning(
            "%d of %d metric results errored (%.0f%%) and are counted as 0.0 - "
            "treat the affected metrics as missing, not as low scores",
            errored,
            total,
            error_rate * 100,
        )

    (REPORTS_DIR / "latest.json").write_text(
        json.dumps(
            {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "summary": summary,
                "cases": [
                    {
                        "golden_id": c.golden_id,
                        "category": c.category,
                        "route": c.route,
                        "abstained": c.abstained,
                        "metrics": [vars(m) for m in c.metrics],
                    }
                    for c in results
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (REPORTS_DIR / "latest.md").write_text(
        render_markdown(results, summary, goldens), encoding="utf-8"
    )

    logger.info("scored %d goldens in %.0fs", len(results), time.perf_counter() - started)
    for name, stats in sorted(summary.items()):
        logger.info(
            "  %-22s mean=%.2f median=%.2f pass=%.0f%%",
            name,
            stats["mean"],
            stats["median"],
            stats["pass_rate"] * 100,
        )

    failures = sum(1 for c in results for m in c.metrics if not m.passed)
    logger.info("wrote %s", REPORTS_DIR / "latest.md")
    if failures:
        logger.warning("%d metric results below threshold", failures)
    return 1 if (failures and args.fail_under) else 0


if __name__ == "__main__":
    sys.exit(main())
