"""Builds the configured embedding provider."""

from __future__ import annotations

import logging

from core.config import Settings

logger = logging.getLogger(__name__)

__all__ = ["build_provider"]


def build_provider(settings: Settings):
    """Instantiates the provider named by ``settings.embedding_provider``."""
    name = (settings.embedding_provider or "minilm").strip().lower()
    if name in ("bge", "bge-m3", "baai/bge-m3"):
        from embeddings.bge import BGEProvider

        return BGEProvider(settings.embedding_model or "BAAI/bge-m3")

    if name not in ("minilm", "all-minilm-l6-v2"):
        logger.warning(f"Unknown EMBEDDING_PROVIDER '{name}', falling back to minilm.")

    from embeddings.minilm import MiniLMProvider

    return MiniLMProvider(settings.embedding_model or "all-MiniLM-L6-v2")
