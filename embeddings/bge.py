"""BAAI/bge-m3 provider (1024 dimensions)."""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

from embeddings.sentence_transformer import SentenceTransformerProvider

__all__ = ["BGEProvider"]


class BGEProvider(SentenceTransformerProvider):
    DIMENSION = 1024

    def __init__(self, model_name: str = "BAAI/bge-m3") -> None:
        super().__init__(model_name, SentenceTransformer, self.DIMENSION)
