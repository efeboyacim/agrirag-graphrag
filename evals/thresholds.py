"""Metric thresholds, and the reasoning behind each number.

**Thresholds are on the aggregate, not per case.** An LLM judge produces outliers
on individual cases no matter how good the system is - one question in thirty
will score 0.0 on a metric because the judge read it oddly that run. Gating CI on
every case would mean a red build most days, and a red build most days is a build
nobody looks at. The mean over the whole set is stable enough to gate on; the
per-case scores still land in the report, which is where you go to diagnose.

Each number is set roughly 0.10 below the measured baseline: far enough below
that ordinary judge variance stays green, close enough that a genuine regression
goes red.

That margin is not a guess. Two consecutive report runs over the *same* cached
agent outputs - so the only thing varying is the judge - moved like this:

    metric                 run 1   run 2   delta
    Faithfulness            0.99    1.00   +0.01
    ContextualRecall        0.91    0.92   +0.01
    Hallucination           0.89    0.90   +0.01
    ContextualRelevancy     0.44    0.43   -0.01
    AnswerRelevancy         0.88    0.91   +0.03
    ContextualPrecision     0.78    0.73   -0.05
    AgronomicUsefulness     0.87    0.93   +0.06

    RoutingAccuracy         1.00    1.00    0.00
    GraphPathRecall         1.00    1.00    0.00
    CorrectAbstention       1.00    1.00    0.00

Judge variance reaches 0.06 on a set this size with nothing changing, so a 0.05
margin would produce false failures. The deterministic three do not move at all,
which is exactly why they are the ones gated per case.

Baselines are from evals/reports/latest.md - regenerate it and revisit these if
the system changes materially.
"""

from typing import Final

# --------------------------------------------------------------------------
# Deterministic metrics. No judge, no variance - these are asserted per case and
# are expected to be perfect.
# --------------------------------------------------------------------------

ROUTING_ACCURACY: Final = 1.0
GRAPH_PATH_RECALL: Final = 1.0
CORRECT_ABSTENTION: Final = 1.0


# --------------------------------------------------------------------------
# Judged metrics. Aggregate means. Baseline -> threshold.
# --------------------------------------------------------------------------

#: baseline mean 0.91. Retrieval finds the facts the reference answer needs.
CONTEXTUAL_RECALL: Final = 0.80

#: baseline mean 0.78. Lower than recall because a traversal returns every row
#: matching the pattern, including rows this particular question did not need.
CONTEXTUAL_PRECISION: Final = 0.65

#: baseline mean 0.44 - by far the lowest score in the suite, and worth
#: understanding rather than explaining away.
#:
#: The metric judges each *statement* inside each context item. Our context items
#: are multi-sentence passages, so a chunk that is exactly the right one still
#: loses points for the sentences in it that do not bear on the question.
#:
#: Measured by context type:
#:     graph rows (one fact each, 165-307 chars)   0.53
#:     corpus chunks (multi-sentence, ~580 chars)  0.39
#:
#: Note the direction: graph retrieval scores *better* here, not worse. The
#: initial assumption was the opposite - that traversal returning many rows would
#: drag relevancy down - and the data contradicted it. The real lever is chunk
#: granularity, so the way to raise this number is smaller chunks, not a
#: different retrieval strategy.
CONTEXTUAL_RELEVANCY: Final = 0.35

#: baseline mean 0.99. The headline safety number: the model stays inside the
#: context it was given. High because synthesis is explicitly instructed to, and
#: because the abstain path removes the cases where it would be tempted not to.
FAITHFULNESS: Final = 0.90

#: baseline mean 0.88.
ANSWER_RELEVANCY: Final = 0.75

#: baseline mean 0.89. DeepEval 4.x scores this as *alignment with ground truth* -
#: higher is better. It previously scored the proportion of violations, where the
#: threshold was a maximum; deepeval emits a deprecation warning saying exactly
#: this. Getting the direction wrong makes a healthy 0.89 look like a total
#: failure, which is precisely what happened on the first run here.
HALLUCINATION: Final = 0.75

#: baseline mean 0.87. Domain metric: would the answer let a farmer act?
AGRONOMIC_USEFULNESS: Final = 0.75
