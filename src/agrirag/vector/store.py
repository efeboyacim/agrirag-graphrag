"""Storage-agnostic vector interface.

Mirrors :mod:`agrirag.graph.store`: the agent depends on this Protocol and never
imports lancedb, so the vector backend is swappable without touching agent logic.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class SearchHit:
    """One retrieved chunk.

    ``score`` is a similarity in [0, 1] where higher is better, normalised by the
    implementation - LanceDB returns a cosine *distance*, and leaking that
    convention into the agent would invert every threshold comparison.
    """

    chunk_id: str
    doc_id: str
    doc_title: str
    heading: str
    text: str
    score: float

    def as_context(self) -> str:
        """Render for the unified ``retrieval_context`` the eval metrics consume.

        Graph facts and text chunks must arrive in the same shape, or DeepEval's
        contextual metrics cannot be applied uniformly across both retrieval paths.
        """
        title = f"{self.doc_title} - {self.heading}" if self.heading else self.doc_title
        return f"[{title}] {self.text}"


@runtime_checkable
class VectorStore(Protocol):
    """Minimal search surface required by the retrieval layer."""

    async def search(self, query: str, k: int = 5) -> list[SearchHit]:
        """Return the ``k`` most similar chunks to ``query``."""
        ...

    async def count(self) -> int:
        """Number of indexed chunks."""
        ...

    async def close(self) -> None:
        """Release resources."""
        ...
