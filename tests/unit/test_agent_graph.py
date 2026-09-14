"""Agent control flow.

The edge functions are pure, so the branching logic - which is where an agent
actually goes wrong - is testable without a database, a model, or a token.
"""

import pytest
from pydantic import ValidationError

from agrirag.agent.graph import _after_grade, _fan_out
from agrirag.agent.nodes import MAX_RETRIES, MIN_RELEVANCE, Sufficiency
from agrirag.agent.state import AgentResult, AgentState, Citation, RouteDecision


def state(**overrides: object) -> AgentState:
    base: dict[str, object] = {
        "question": "q",
        "trace_id": "t",
        "route": "graph",
        "retries": 0,
        "retrieval_context": [],
        "graph_rows": [],
        "chunks": [],
    }
    base.update(overrides)
    return base  # type: ignore[return-value]


# --------------------------------------------------------------------------
# Fan-out
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("route", "expected"),
    [
        ("graph", ["graph_retrieve"]),
        ("semantic", ["semantic_retrieve"]),
        ("both", ["graph_retrieve", "semantic_retrieve"]),
    ],
)
def test_fan_out_maps_route_to_nodes(route: str, expected: list[str]) -> None:
    assert _fan_out(state(route=route)) == expected


def test_both_returns_two_nodes_so_they_run_in_one_superstep() -> None:
    """Returning a list is what makes LangGraph run them concurrently. A string
    would serialise them and quietly remove the parallelism."""
    result = _fan_out(state(route="both"))

    assert isinstance(result, list)
    assert len(result) == 2


def test_fan_out_defaults_to_both_when_routing_is_missing() -> None:
    """If the router failed, running both paths still answers the question."""
    assert _fan_out(state(route=None)) == ["graph_retrieve", "semantic_retrieve"]


# --------------------------------------------------------------------------
# Post-grade branching
# --------------------------------------------------------------------------


def test_sufficient_context_goes_to_synthesis() -> None:
    assert _after_grade(state(sufficient=True, retrieval_context=["fact"])) == "synthesize"


def test_a_thin_graph_result_retries_through_semantic_search() -> None:
    """The cycle in the graph: a graph-only question that came back thin may
    still be explained by the corpus."""
    decision = _after_grade(
        state(sufficient=False, route="graph", retries=1, retrieval_context=["thin"])
    )

    assert decision == "semantic_retrieve"


def test_the_retry_happens_at_most_once() -> None:
    assert (
        _after_grade(
            state(
                sufficient=False,
                route="graph",
                retries=MAX_RETRIES + 1,
                retrieval_context=["thin"],
            )
        )
        == "abstain"
    )


def test_a_semantic_question_does_not_retry_into_itself() -> None:
    """Re-running the same search with a wider k on a question that already
    failed the relevance floor just burns a request."""
    assert _after_grade(state(sufficient=False, route="semantic", retries=1)) == "abstain"


def test_verified_graph_facts_are_answered_even_when_graded_incomplete() -> None:
    """A compound question - "which treatments, and are any restricted?" - often
    has half its answer in the graph. Refusing while holding rows a traversal
    verified is worse than answering and saying what is missing, which the
    synthesis prompt already instructs.
    """
    decision = _after_grade(
        state(
            sufficient=False,
            route="graph",
            retries=MAX_RETRIES + 1,
            graph_rows=[{"input_name": "Spinosad"}],
            retrieval_context=["Maize is treated by Spinosad."],
        )
    )

    assert decision == "synthesize"


def test_chunks_alone_the_grader_rejected_are_not_enough_to_answer() -> None:
    """Chunks passing the relevance floor is not the same as answering the question.

    The evaluation found this: "what will the wheat price be next season?" pulled
    one sunn-pest chunk past the floor, the grader correctly said it could not
    answer the question, and an earlier rule synthesised anyway. The model wrote
    an honest "the context does not cover this" - but never set `abstained`, so
    no metric could see that it had declined.

    Graph rows are the distinction. They are facts a traversal verified; chunks
    the grader has already rejected are not evidence of anything.
    """
    decision = _after_grade(
        state(
            sufficient=False,
            route="semantic",
            retries=MAX_RETRIES + 1,
            chunks=[{"chunk_id": "c1"}],
            retrieval_context=["Some tangentially related passage."],
        )
    )

    assert decision == "abstain"


def test_nothing_retrieved_abstains() -> None:
    """The off-domain case. The relevance floor is what empties the context here:
    without it vector search returns its k nearest neighbours however far away
    they are, and this branch would never be reached."""
    decision = _after_grade(
        state(
            sufficient=False,
            route="semantic",
            retries=MAX_RETRIES + 1,
            graph_rows=[],
            chunks=[],
            retrieval_context=[],
        )
    )

    assert decision == "abstain"


# --------------------------------------------------------------------------
# Contracts
# --------------------------------------------------------------------------


def test_the_relevance_floor_sits_between_the_measured_populations() -> None:
    """Off-domain questions score 0.45-0.55 against this corpus, in-domain
    0.78-0.84. A floor outside that gap breaks either abstention or recall."""
    assert 0.56 < MIN_RELEVANCE < 0.77


def test_the_grader_fails_open_on_an_empty_response() -> None:
    """The model intermittently returns a tool call with no arguments. With
    context in hand, answering beats refusing over a transport glitch."""
    assert Sufficiency().sufficient is True


def test_route_decision_defaults_are_conservative() -> None:
    decision = RouteDecision(route="graph", reason="because")

    assert decision.template is None
    assert decision.organic_only is False


def test_agent_result_separates_answer_from_retrieval_context() -> None:
    """Phase 5's generation metrics read `answer`; retrieval metrics read
    `retrieval_context`. Collapsing them would make a retrieval failure
    indistinguishable from a synthesis failure."""
    result = AgentResult.from_state(
        state(
            answer="the answer",
            route="both",
            route_reason="why",
            retrieval_context=["fact one", "fact two"],
            graph_rows=[{"a": 1}],
            chunks=[{"chunk_id": "c1"}],
            citations=[{"kind": "graph", "ref": "crop_maize", "label": "Maize"}],
        )
    )

    assert result.answer == "the answer"
    assert result.retrieval_context == ["fact one", "fact two"]
    assert result.graph_row_count == 1
    assert result.chunk_count == 1
    assert result.citations == [Citation(kind="graph", ref="crop_maize", label="Maize")]


def test_agent_result_survives_a_state_that_never_reached_synthesis() -> None:
    result = AgentResult.from_state(state(abstained=True, answer="I don't know"))

    assert result.abstained is True
    assert result.citations == []


# --------------------------------------------------------------------------
# RouteDecision tolerance
#
# Smaller models fill this schema loosely. These coercions only fire when the
# raw value is not already valid, so a model that answers correctly is
# unaffected - but without them, running against a local llama3.2 failed every
# routing call and the agent ran both retrievers on every question.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("graph", "graph"),
        ("Graph", "graph"),
        ("  SEMANTIC  ", "semantic"),
        ("both", "both"),
        ("use the graph", "graph"),
        ("semantic search is best here", "semantic"),
        ("graph and semantic", "both"),
    ],
)
def test_route_accepts_what_models_actually_emit(raw: str, expected: str) -> None:
    assert RouteDecision(route=raw, reason="r").route == expected  # type: ignore[arg-type]


def test_an_unreadable_route_still_fails() -> None:
    """Coercion must not become guessing. A value with no recognisable path
    should fail so the node's fallback logs why, rather than silently routing
    somewhere arbitrary."""
    with pytest.raises(ValidationError):
        RouteDecision(route="telepathy", reason="r")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("yes", True), ("true", True), ("1", True), ("no", False), ("false", False), (None, False)],
)
def test_organic_flag_accepts_loose_booleans(raw: object, expected: bool) -> None:
    decision = RouteDecision(route="graph", reason="r", organic_only=raw)  # type: ignore[arg-type]

    assert decision.organic_only is expected


@pytest.mark.parametrize("raw", ["", "none", "N/A"])
def test_a_blank_template_becomes_none(raw: str) -> None:
    assert RouteDecision(route="graph", reason="r", template=raw).template is None
