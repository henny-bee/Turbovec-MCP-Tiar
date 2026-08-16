from typing import List, Dict, Any, Protocol


class Reranker(Protocol):
    def rerank(
        self, query: str, candidates: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        """Reranks candidates based on semantic relevance to query."""
        ...
