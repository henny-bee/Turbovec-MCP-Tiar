import logging
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from embeddings.base import EmbeddingProvider
from memory.errors import SearchError, EMBEDDING_FAILED

logger = logging.getLogger(__name__)

class MiniLMProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        logger.info(f"Initializing SentenceTransformer: {model_name}")
        try:
            self._model = SentenceTransformer(model_name)
        except Exception as e:
            raise SearchError(
                code=EMBEDDING_FAILED,
                message=f"Failed to load embedding model {model_name}: {e}",
                subsystem="EMBEDDING",
                retry_safe=True
            )

    @property
    def dimension(self) -> int:
        return 384

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        try:
            embeddings = self._model.encode(texts)
            if isinstance(embeddings, np.ndarray):
                return embeddings.tolist()
            return embeddings
        except Exception as e:
            logger.error(f"Embedding failed using {self.model_name}: {e}")
            raise SearchError(
                code=EMBEDDING_FAILED,
                message=f"Embedding generation failed: {e}",
                subsystem="EMBEDDING",
                retry_safe=True
            )
