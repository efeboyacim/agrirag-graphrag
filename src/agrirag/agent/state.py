"""Agent state and the result contract.

The reducers here are load-bearing. When the router fans out to both retrievers
they execute in the same LangGraph superstep, and ``operator.add`` merges their
outputs into one ``retrieval_context`` without either branch knowing the other
exists. Without the reducers, the second branch to finish would overwrite the
first - silently, and only on "both" questions.

State is kept to plain dicts and primitives rather than dataclasses. The SQLite
checkpointer serialises every superstep, and plain data keeps that cheap and
avoids a class of surprises where a resumed thread deserialises into something
subtly different.
"""

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field, field_validator

Route = Literal["graph", "semantic", "both"]


class AgentState(TypedDict, total=False):
    """Everything that flows between nodes."""

    # -- input ------------------------------------------------------------
    question: str
    trace_id: str

    # -- routing ----------------------------------------------------------
    #: Set by the caller to skip the routing node entirely. Exists so the API can
    #: force 'semantic' on a multi-hop question and show, side by side, what the
    #: graph traversal actually contributes.
    forced_route: Route | None
    route: Route
    route_reason: str
    template: str | None
    organic_only: bool

    # -- retrieval --------------------------------------------------------
    linked_entities: list[dict[str, Any]]
    cypher: str | None
    graph_rows: Annotated[list[dict[str, Any]], operator.add]
    chunks: Annotated[list[dict[str, Any]], operator.add]
    #: The unified, eval-facing context. Graph facts arrive linearised into
    #: sentences so that both retrieval paths produce the same shape.
    retrieval_context: Annotated[list[str], operator.add]

    # -- grading ----------------------------------------------------------
    sufficient: bool
    sufficiency_reason: str
    retries: int

    # -- output -----------------------------------------------------------
    answer: str
    citations: list[dict[str, str]]
    abstained: bool


class Citation(BaseModel):
    """Where one piece of the answer came from."""

    kind: Literal["graph", "document"]
    ref: str = Field(description="Node id for graph facts, chunk id for documents.")
    label: str = Field(description="Human-readable source name.")


class AgentResult(BaseModel):
    """The agent's contract with the API and the evaluation suite.

    ``answer`` and ``retrieval_context`` are deliberately separate fields:
    Phase 5's generation metrics consume only the former, retrieval metrics only
    the latter. Collapsing them would make it impossible to tell a retrieval
    failure from a synthesis failure.
    """

    question: str
    answer: str
    route: Route
    route_reason: str
    retrieval_context: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    abstained: bool = False

    # -- traceability -----------------------------------------------------
    trace_id: str = ""
    template: str | None = None
    cypher: str | None = None
    linked_entities: list[dict[str, Any]] = Field(default_factory=list)
    graph_row_count: int = 0
    chunk_count: int = 0
    retries: int = 0

    @classmethod
    def from_state(cls, state: AgentState) -> "AgentResult":
        """Project the final graph state into the public result."""
        return cls(
            question=state.get("question", ""),
            answer=state.get("answer", ""),
            route=state.get("route", "both"),
            route_reason=state.get("route_reason", ""),
            retrieval_context=state.get("retrieval_context", []),
            citations=[Citation.model_validate(c) for c in state.get("citations", [])],
            abstained=state.get("abstained", False),
            trace_id=state.get("trace_id", ""),
            template=state.get("template"),
            cypher=state.get("cypher"),
            linked_entities=state.get("linked_entities", []),
            graph_row_count=len(state.get("graph_rows", [])),
            chunk_count=len(state.get("chunks", [])),
            retries=state.get("retries", 0),
        )


_TRUEISH = {"true", "yes", "y", "1", "t"}
_FALSEISH = {"false", "no", "n", "0", "f", "none", "null", ""}


class RouteDecision(BaseModel):
    """Structured output of the routing node.

    The validators below exist because smaller models fill this schema loosely.
    Running against a local llama3.2, ``organic_only`` came back as the string
    "no" and ``route`` as prose containing the word rather than the bare literal -
    both rejected by strict Pydantic, so every routing call failed and the agent
    fell back to running both retrievers on every question.

    They only fire when the raw value is *not* already valid, so a model that
    fills the schema correctly is unaffected. Coercing here rather than loosening
    the field types keeps the contract strict for everything downstream.
    """

    route: Route = Field(
        description=(
            "'graph' for questions about how entities relate (which pest attacks "
            "what, what treats it, what is restricted where). 'semantic' for "
            "definitional or explanatory questions answered by prose. 'both' when "
            "the question needs facts and an explanation of why."
        )
    )
    reason: str = Field(description="One sentence explaining the choice.")
    template: str | None = Field(
        default=None,
        description=(
            "Name of the graph query template to use. Required when route is graph or both."
        ),
    )
    organic_only: bool = Field(
        default=False,
        description="True only when the question explicitly asks for organic-approved inputs.",
    )

    @field_validator("route", mode="before")
    @classmethod
    def _normalise_route(cls, value: object) -> object:
        """Accept 'Graph', ' semantic ', or a sentence containing the word."""
        if not isinstance(value, str):
            return value
        cleaned = value.strip().lower()
        if cleaned in {"graph", "semantic", "both"}:
            return cleaned
        # A model that wrote a sentence instead of a literal still told us which
        # path it meant. "both" is checked first: "graph and semantic" means both.
        if "both" in cleaned or ("graph" in cleaned and "semantic" in cleaned):
            return "both"
        if "graph" in cleaned:
            return "graph"
        if "semantic" in cleaned:
            return "semantic"
        # Genuinely unreadable - let it fail so the node's fallback logs why.
        return value

    @field_validator("organic_only", mode="before")
    @classmethod
    def _normalise_flag(cls, value: object) -> object:
        """Accept 'no', 'false', 0 and friends as booleans."""
        if isinstance(value, str):
            cleaned = value.strip().lower()
            if cleaned in _TRUEISH:
                return True
            if cleaned in _FALSEISH:
                return False
        if value is None:
            return False
        return value

    @field_validator("template", mode="before")
    @classmethod
    def _blank_template_is_none(cls, value: object) -> object:
        """Treat "", "none" and "null" as no template chosen."""
        if isinstance(value, str) and value.strip().lower() in {"", "none", "null", "n/a"}:
            return None
        return value
