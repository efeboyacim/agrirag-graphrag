"""Local embeddings.

Runs BAAI/bge-small-en-v1.5 as ONNX in-process via fastembed. No API key, no
network after the first model download, and identical vectors on every run - so
re-indexing the corpus costs nothing and the Phase 5 retrieval metrics are not
measuring embedding-API drift.

The model is ~130 MB and loads lazily on first use, which keeps process start and
the unit suite fast.
"""

import logging
from functools import lru_cache
from typing import TYPE_CHECKING

from agrirag.config import get_settings

if TYPE_CHECKING:  # pragma: no cover - import cost is the whole point of deferring
    from fastembed import TextEmbedding

logger = logging.getLogger(__name__)

#: BGE models are trained with an asymmetric objective: queries get a prefix that
#: passages do not. Omitting it measurably degrades recall, and it is a silent
#: failure - retrieval still "works", just worse.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class LocalEmbedder:
    """Embeds text with a local ONNX model."""

    def __init__(self, model_name: str | None = None, dim: int | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self.dim = dim or settings.embedding_dim
        self.cache_dir = settings.embedding_cache_dir
        self._model: TextEmbedding | None = None

    def _ensure_model(self) -> "TextEmbedding":
        if self._model is None:
            from fastembed import TextEmbedding

            logger.info(
                "loading embedding model %s (cache=%s)",
                self.model_name,
                self.cache_dir or "library default",
            )
            self._model = (
                TextEmbedding(model_name=self.model_name, cache_dir=self.cache_dir)
                if self.cache_dir
                else TextEmbedding(model_name=self.model_name)
            )
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed corpus passages for indexing."""
        model = self._ensure_model()
        return [vector.tolist() for vector in model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        """Embed a search query, with the asymmetric BGE prefix applied."""
        model = self._ensure_model()
        vector = next(iter(model.embed([QUERY_PREFIX + text])))
        return list(vector.tolist())


@lru_cache
def get_embedder() -> LocalEmbedder:
    """Process-wide embedder. Cached so the ONNX model is loaded once."""
    return LocalEmbedder()
