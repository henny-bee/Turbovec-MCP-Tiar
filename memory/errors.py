from dataclasses import dataclass


@dataclass
class SearchError(Exception):
    code: str
    message: str
    subsystem: str
    retry_safe: bool

    def __str__(self) -> str:
        return f"[{self.subsystem}] {self.code}: {self.message} (Retry Safe: {self.retry_safe})"


# Error codes
MEMORY_LAYER_DEGRADED = "MEMORY_LAYER_DEGRADED"
VECTOR_INDEX_UNAVAILABLE = "VECTOR_INDEX_UNAVAILABLE"
VECTOR_INDEX_CORRUPTED = "VECTOR_INDEX_CORRUPTED"
SQLITE_UNAVAILABLE = "SQLITE_UNAVAILABLE"
SQLITE_TRANSACTION_FAILED = "SQLITE_TRANSACTION_FAILED"
EMBEDDING_FAILED = "EMBEDDING_FAILED"
RERANKER_FAILED = "RERANKER_FAILED"
ONTOLOGY_VALIDATION_FAILED = "ONTOLOGY_VALIDATION_FAILED"
TEMPORAL_QUERY_FAILED = "TEMPORAL_QUERY_FAILED"
