from typing import List, Protocol


class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int:
        """Returns the dimension of generated embeddings."""
        ...

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of texts."""
        ...
