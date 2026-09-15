"""all-MiniLM-L6-v2 provider (384 dimensions, the default)."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from embeddings.sentence_transformer import SentenceTransformerProvider

__all__ = ["MiniLMProvider"]


class MiniLMProvider(SentenceTransformerProvider):
    DIMENSION = 384

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name, SentenceTransformer, self.DIMENSION)
