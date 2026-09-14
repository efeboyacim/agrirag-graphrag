"""Generation quality - the CI gate.

These metrics read the answer. They never look at whether retrieval found the
right material; that is the retrieval suite's job. Keeping them apart is what
makes a failure attributable: low faithfulness with healthy contextual recall
means the model invented something over good context, while low scores in both
point at retrieval.

Same shape as the retrieval suite - deterministic checks per case, judged metrics
asserted on the aggregate and read from the report so they are paid for once.
"""

import pytest

from evals import thresholds
from evals.cases import build_case, scoreable
from evals.dataset import Golden, active_goldens
from evals.metrics import CorrectAbstention
from evals.runner import load_or_fail
from evals.test_retrieval import assert_mean_at_least, report, runs  # noqa: F401

ALL_GOLDENS = active_goldens()
SCOREABLE = scoreable(ALL_GOLDENS)
ABSTAIN_GOLDENS = [g for g in ALL_GOLDENS if g.must_abstain]

__all__ = ["load_or_fail"]


def _ids(goldens: list[Golden]) -> list[str]:
    return [g.id for g in goldens]


# --------------------------------------------------------------------------
# Deterministic - per case
# --------------------------------------------------------------------------


@pytest.mark.parametrize("golden", ALL_GOLDENS, ids=_ids(ALL_GOLDENS))
def test_correct_abstention(golden: Golden, runs: dict) -> None:  # noqa: F811
    """Did the agent decline exactly when it should have?

    Penalises both directions, which matters more than it sounds. A hallucination
    metric alone would happily reward a system that refuses everything - refusing
    never hallucinates - so something has to make over-caution costly too.
    """
    metric = CorrectAbstention(golden, threshold=thresholds.CORRECT_ABSTENTION)
    metric.measure(build_case(golden, runs[golden.id]))

    assert metric.is_successful(), metric.reason


@pytest.mark.parametrize("golden", ABSTAIN_GOLDENS, ids=_ids(ABSTAIN_GOLDENS))
def test_declining_answers_say_so_plainly(golden: Golden, runs: dict) -> None:  # noqa: F811
    """A refusal has to read as a refusal, not as a confident non-answer.

    Note what is *not* asserted: that retrieval_context is empty. Abstention does
    not require it. "What will the wheat price be next season?" pulls a sunn-pest
    chunk past the relevance floor - the word "wheat" is genuinely in it - and the
    agent still correctly declines, because the grader judged that context unable
    to answer the question and no graph facts backed it.

    The invariant that matters is that no *verified* facts were held, which is
    what separates declining from the compound-question case where the agent
    answers half and flags the rest.
    """
    run = runs[golden.id]

    assert run.abstained
    assert "don't have enough information" in run.answer
    assert run.graph_row_count == 0


# --------------------------------------------------------------------------
# Judged - aggregate
# --------------------------------------------------------------------------


def test_faithfulness(report: dict) -> None:  # noqa: F811
    """Is every claim in the answer supported by the retrieved context?

    The headline safety metric for a RAG system: it measures whether the model
    stayed inside what it was given.
    """
    assert_mean_at_least(report, "Faithfulness", thresholds.FAITHFULNESS)


def test_answer_relevancy(report: dict) -> None:  # noqa: F811
    """Does the answer address the question that was asked?"""
    assert_mean_at_least(report, "AnswerRelevancy", thresholds.ANSWER_RELEVANCY)


def test_hallucination(report: dict) -> None:  # noqa: F811
    """Does the answer align with hand-written ground truth?

    Scored against the golden's ``context``, not ``retrieval_context``. This is
    the only metric that reads ground truth the system never saw: faithfulness
    asks "did you stay inside what you were given", this asks "was what you were
    given actually true". An answer can be perfectly faithful to bad context and
    still be false.

    Higher is better in DeepEval 4.x, which reversed the direction from earlier
    releases - see :mod:`evals.thresholds`.
    """
    assert_mean_at_least(report, "Hallucination", thresholds.HALLUCINATION)


def test_agronomic_usefulness(report: dict) -> None:  # noqa: F811
    """Would the answer let a farmer act?

    A domain metric, because the generic ones miss the thing that matters. An
    answer can be faithful, relevant and non-hallucinatory while still being
    useless - "several treatments exist for this pest" is all three. Application
    rates and pre-harvest intervals are what make it actionable, and the
    pre-harvest interval is a legal constraint on when the crop can be harvested.
    """
    assert_mean_at_least(report, "AgronomicUsefulness", thresholds.AGRONOMIC_USEFULNESS)
