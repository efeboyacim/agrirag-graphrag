"""Deterministic, non-LLM metrics.

These matter more than their simplicity suggests. Every other metric in the suite
is graded by a language model, which means it is slow, costs money, and moves
between runs even when the code does not. These three are free, instant and
exactly reproducible, so they can gate every CI run - and when an LLM-graded score
drops, they are how you tell a real regression from judge variance.

They also isolate failures the judged metrics cannot attribute. A low faithfulness
score says the answer was not grounded; it does not say whether retrieval fetched
the wrong facts or the router sent the question down the wrong path. RoutingAccuracy
and GraphPathRecall answer exactly that.
"""

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from evals.dataset import Golden


class _DeterministicMetric(BaseMetric):
    """Shared plumbing. DeepEval expects `measure`, `a_measure` and `is_successful`."""

    def __init__(self, golden: Golden, threshold: float = 1.0) -> None:
        self.golden = golden
        self.threshold = threshold
        self.score = 0.0
        self.reason = ""
        self.success = False
        # No model involved, so nothing to run concurrently and nothing to spend.
        self.async_mode = False
        self.evaluation_cost = 0.0

    async def a_measure(self, test_case: LLMTestCase, *args: object, **kwargs: object) -> float:
        return self.measure(test_case, *args, **kwargs)

    def is_successful(self) -> bool:
        return self.success


class RoutingAccuracy(_DeterministicMetric):
    """Did the router choose an acceptable retrieval path?

    Attributes a whole class of failure that no output-based metric can see: an
    answer assembled from the wrong store may still read well, and a judged metric
    will happily score it highly.

    Acceptable rather than exact, deliberately. A "both" question whose facts
    happen to be written down in prose can be answered correctly by semantic
    search alone, and marking that wrong would punish a good answer for taking a
    cheaper path.
    """

    __name__ = "RoutingAccuracy"

    def measure(self, test_case: LLMTestCase, *args: object, **kwargs: object) -> float:
        actual = str((test_case.metadata or {}).get("route", ""))

        if self.golden.must_abstain:
            abstained = bool((test_case.metadata or {}).get("abstained"))
            self.success = abstained
            self.score = 1.0 if abstained else 0.0
            self.reason = (
                "correctly declined to answer"
                if abstained
                else f"should have abstained but answered via '{actual}'"
            )
            return self.score

        self.success = self.golden.route_is_acceptable(actual)
        self.score = 1.0 if self.success else 0.0
        expected = self.golden.acceptable_routes or (self.golden.expected_route,)
        self.reason = (
            f"routed to '{actual}' as expected"
            if self.success
            else f"routed to '{actual}', expected one of {list(expected)}"
        )
        return self.score


class GraphPathRecall(_DeterministicMetric):
    """Did entity linking resolve the nodes the traversal needed?

    Scored as the fraction of expected ids that were actually linked. This is the
    single most diagnostic number in the suite: entity linking is upstream of
    everything on the graph path, so when it misses, retrieval returns nothing and
    every downstream metric collapses for a reason that has nothing to do with the
    graph, the query or the model.
    """

    __name__ = "GraphPathRecall"

    def __init__(self, golden: Golden, threshold: float = 1.0) -> None:
        super().__init__(golden, threshold)

    def measure(self, test_case: LLMTestCase, *args: object, **kwargs: object) -> float:
        expected = set(self.golden.expected_entity_ids)
        if not expected:
            # Nothing to resolve - a semantic or abstain case. Vacuously satisfied;
            # scoring it 0 would drag the aggregate down for cases the metric does
            # not apply to.
            self.score = 1.0
            self.success = True
            self.reason = "no entities expected"
            return self.score

        # Entity linking only runs inside graph_retrieve. If the router chose
        # semantic-only, there is nothing for this metric to measure - and
        # RoutingAccuracy already covers "should have used the graph and did
        # not", so scoring zero here would double-penalise one mistake.
        if str((test_case.metadata or {}).get("route", "")) == "semantic":
            self.score = 1.0
            self.success = True
            self.reason = "graph path not taken; entity linking not applicable"
            return self.score

        linked = {
            str(entity.get("id"))
            for entity in (test_case.metadata or {}).get("linked_entities", [])
        }
        found = expected & linked

        self.score = len(found) / len(expected)
        self.success = self.score >= self.threshold
        missing = sorted(expected - linked)
        self.reason = (
            f"resolved all of {sorted(expected)}"
            if not missing
            else f"missing {missing}; resolved {sorted(linked)}"
        )
        return self.score


class CorrectAbstention(_DeterministicMetric):
    """Did the agent decline exactly when it should have?

    Penalises both directions. Answering an unanswerable question is the
    hallucination case; declining an answerable one is the over-cautious case that
    a hallucination metric alone would happily reward, because a system that
    refuses everything never hallucinates.
    """

    __name__ = "CorrectAbstention"

    def measure(self, test_case: LLMTestCase, *args: object, **kwargs: object) -> float:
        abstained = bool((test_case.metadata or {}).get("abstained"))
        correct = abstained == self.golden.must_abstain

        self.score = 1.0 if correct else 0.0
        self.success = correct
        if correct:
            self.reason = "declined as expected" if abstained else "answered as expected"
        else:
            self.reason = (
                "declined a question it should have answered"
                if abstained
                else "answered a question it should have declined"
            )
        return self.score
