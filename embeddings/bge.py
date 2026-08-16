import logging
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from embeddings.base import EmbeddingProvider
from memory.errors import SearchError, EMBEDDING_FAILED

logger = logging.getLogger(__name__)

class BGEProvider(EmbeddingProvider):
    def __init__(self, model_name: str = "BAAI/bge-m3"):
        self.model_name = model_name
        logger.info(f"Initializing SentenceTransformer: {model_name}")
        try:
            self._model = SentenceTransformer(model_name)
        except Exception as e:
            raise SearchError(
                code=EMBEDDING_FAILED,
                message=f"Failed to load embedding model {model_name}: {e}. Ensure sentence-transformers is installed and model is accessible.",
                subsystem="EMBEDDING",
                retry_safe=True
            )

    @property
    def dimension(self) -> int:
        # BGE-M3 has a default dimension of 1024
        return 1024

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
