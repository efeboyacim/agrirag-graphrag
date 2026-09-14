"""Turning agent runs into DeepEval test cases.

The mapping here is what makes the retrieval/generation split real, so it is
worth being explicit about which field feeds which metric:

    input             -> the question
    actual_output     -> the agent's answer            (generation metrics)
    retrieval_context -> what retrieval actually found (retrieval metrics)
    expected_output   -> the reference answer          (recall/precision)
    context           -> hand-authored ground truth    (hallucination only)

``context`` and ``retrieval_context`` are different things and DeepEval treats
them differently. ``retrieval_context`` is what the system fetched;
``context`` is what is true. HallucinationMetric compares the answer against
what is *true*, which is why the goldens carry both.
"""

from deepeval.test_case import LLMTestCase

from evals.dataset import Golden
from evals.runner import AgentRun


def build_case(golden: Golden, run: AgentRun) -> LLMTestCase:
    """Build a DeepEval test case from a golden and the agent's run of it.

    The traceability fields ride along in ``metadata`` so the deterministic
    metrics can read the route and the linked entities. DeepEval ignores
    metadata when scoring, so this costs nothing.
    """
    return LLMTestCase(
        name=golden.id,
        input=golden.input,
        actual_output=run.answer,
        expected_output=golden.expected_output,
        retrieval_context=list(run.retrieval_context) or None,
        context=list(golden.context) or None,
        metadata={
            "golden_id": golden.id,
            "category": golden.category,
            "route": run.route,
            "route_reason": run.route_reason,
            "abstained": run.abstained,
            "linked_entities": run.linked_entities,
            "template": run.template,
            "graph_row_count": run.graph_row_count,
            "chunk_count": run.chunk_count,
            "hops": golden.hops,
        },
    )


def scoreable(goldens: list[Golden]) -> list[Golden]:
    """Goldens the LLM-graded metrics can actually score.

    Abstention cases are excluded. They have no retrieval context by design, so
    a contextual metric has nothing to measure and a faithfulness score over an
    empty context is meaningless rather than zero. Correct declining is measured
    by :class:`~evals.metrics.CorrectAbstention` instead - which is the point of
    having deterministic metrics alongside the judged ones.
    """
    answerable = [g for g in goldens if not g.must_abstain]
    return answerable
