"""Retrieval quality - the CI gate.

These metrics read ``retrieval_context`` and never look at the answer. A failure
here means the system fetched the wrong facts, which is a different problem from
writing a bad answer over the right ones and needs a different fix.

Two kinds of test, deliberately different in shape:

* **Deterministic, per case.** Routing and entity linking are computed without a
  model, so they are free, instant and exactly reproducible. Every case must
  pass. These gate a pull request with no API key and no spend.

* **Judged, aggregate.** Scored by an LLM, so individual cases are noisy - one
  question in thirty will score oddly on any given run. The mean over the set is
  what gets asserted; per-case scores live in the report for diagnosis.

The judged assertions read the report rather than re-scoring, so the metrics are
paid for exactly once:

    uv run python -m evals.runner --refresh   # run the agent over the goldens
    uv run python -m evals.report             # score once, write the report
    uv run pytest evals/                      # gate on it
"""

import json
import statistics
from pathlib import Path

import pytest

from evals import thresholds
from evals.cases import build_case, scoreable
from evals.dataset import Golden, active_goldens
from evals.metrics import GraphPathRecall, RoutingAccuracy
from evals.report import REPORTS_DIR
from evals.runner import load_or_fail

ALL_GOLDENS = active_goldens()
SCOREABLE = scoreable(ALL_GOLDENS)


@pytest.fixture(scope="session")
def runs() -> dict:
    return load_or_fail(ALL_GOLDENS)


@pytest.fixture(scope="session")
def report() -> dict:
    """Per-case scores from the last report run."""
    path: Path = REPORTS_DIR / "latest.json"
    if not path.exists():
        pytest.skip(f"no report at {path} - run: uv run python -m evals.report")
    return json.loads(path.read_text(encoding="utf-8"))


def metric_scores(report: dict, name: str) -> list[float]:
    return [m["score"] for c in report["cases"] for m in c["metrics"] if m["name"] == name]


def assert_mean_at_least(report: dict, name: str, threshold: float) -> None:
    scores = metric_scores(report, name)
    assert scores, f"{name} was not scored - is the report stale?"

    mean = statistics.mean(scores)
    worst = sorted(scores)[:3]
    assert mean >= threshold, (
        f"{name} mean {mean:.2f} below threshold {threshold:.2f} "
        f"(n={len(scores)}, worst: {[round(s, 2) for s in worst]}). "
        "See evals/reports/latest.md for per-case reasons."
    )


def _ids(goldens: list[Golden]) -> list[str]:
    return [g.id for g in goldens]


# --------------------------------------------------------------------------
# Deterministic - free, instant, per case
# --------------------------------------------------------------------------


@pytest.mark.parametrize("golden", ALL_GOLDENS, ids=_ids(ALL_GOLDENS))
def test_routing_accuracy(golden: Golden, runs: dict) -> None:
    """Did the router pick an acceptable retrieval path?

    Catches a failure no output-based metric can see: an answer assembled from
    the wrong store often still reads well, and a judged metric will score it
    highly.
    """
    metric = RoutingAccuracy(golden, threshold=thresholds.ROUTING_ACCURACY)
    metric.measure(build_case(golden, runs[golden.id]))

    assert metric.is_successful(), metric.reason


@pytest.mark.parametrize("golden", ALL_GOLDENS, ids=_ids(ALL_GOLDENS))
def test_graph_path_recall(golden: Golden, runs: dict) -> None:
    """Did entity linking resolve the nodes the traversal needed?

    The most diagnostic check in the suite. Entity linking is upstream of
    everything on the graph path, so when it misses, retrieval returns nothing
    and every downstream metric collapses for a reason that has nothing to do
    with the graph, the query or the model. It caught two real bugs during
    development: a label missing from the fulltext index, and Neo4j's analyzer
    not stemming, so "grapes" never matched the stored name "Grape".
    """
    metric = GraphPathRecall(golden, threshold=thresholds.GRAPH_PATH_RECALL)
    metric.measure(build_case(golden, runs[golden.id]))

    assert metric.is_successful(), metric.reason


@pytest.mark.parametrize("golden", SCOREABLE, ids=_ids(SCOREABLE))
def test_answerable_questions_retrieved_something(golden: Golden, runs: dict) -> None:
    """A cheap guard that catches total retrieval failure before any judged
    metric is paid for."""
    run = runs[golden.id]

    assert run.retrieval_context, f"{golden.id} retrieved nothing"
    assert not run.abstained, f"{golden.id} declined an answerable question"


# --------------------------------------------------------------------------
# Judged - aggregate
# --------------------------------------------------------------------------


def test_contextual_recall(report: dict) -> None:
    """Did retrieval find the facts the reference answer relies on?

    The most consequential of the three: a low score means the information never
    reached the model, so no amount of work on the synthesis prompt can fix it.
    """
    assert_mean_at_least(report, "ContextualRecall", thresholds.CONTEXTUAL_RECALL)


def test_contextual_precision(report: dict) -> None:
    """Are the relevant items ranked above the irrelevant ones?"""
    assert_mean_at_least(report, "ContextualPrecision", thresholds.CONTEXTUAL_PRECISION)


def test_contextual_relevancy(report: dict) -> None:
    """Is the retrieved material on topic, statement by statement?

    The lowest score in the suite, and the threshold reflects the measured
    baseline rather than an aspiration. See :mod:`evals.thresholds` for why it
    sits where it does - short version: the metric judges each statement inside
    each context item, and multi-sentence chunks lose points for the sentences
    that do not bear on the question.
    """
    assert_mean_at_least(report, "ContextualRelevancy", thresholds.CONTEXTUAL_RELEVANCY)
