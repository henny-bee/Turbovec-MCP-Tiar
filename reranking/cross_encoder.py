import logging
import time
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from reranking.base import Reranker
from memory.errors import SearchError, RERANKER_FAILED

logger = logging.getLogger(__name__)


class CrossEncoderReranker(Reranker):
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        enabled: bool = True,
    ):
        self.model_name = model_name
        self.enabled = enabled
        self._model = None

    def _load_model(self) -> None:
        if self._model is None and self.enabled:
            logger.info(f"Loading CrossEncoder model: {self.model_name}")
            try:
                self._model = CrossEncoder(self.model_name)
            except Exception as e:
                logger.error(
                    f"Failed to load CrossEncoder model {self.model_name}: {e}"
                )
                self.enabled = False  # disable so we don't try reloading constantly
                raise SearchError(
                    code=RERANKER_FAILED,
                    message=f"Failed to load CrossEncoder {self.model_name}: {e}",
                    subsystem="RERANKER",
                    retry_safe=True,
                )

    def rerank(
        self, query: str, candidates: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        if not self.enabled or not candidates:
            return candidates[:top_k]

        try:
            self._load_model()
        except Exception:
            # Gracefully fallback if loading failed but don't crash entirely if disabled
            return candidates[:top_k]

        if self._model is None:
            return candidates[:top_k]

        start_time = time.perf_counter()
        try:
            # Form pairs: [query, candidate_text]
            pairs = []
            for c in candidates:
                # build descriptive text for reranking
                properties = c.get("properties", {})
                desc = (
                    properties.get("description", "")
                    if isinstance(properties, dict)
                    else str(properties)
                )
                obs_text = " ".join(
                    [o.get("content", "") for o in c.get("observations", [])]
                )
                text_repr = f"{c.get('name', '')} {c.get('node_type', '')} {desc} {obs_text}".strip()
                pairs.append([query, text_repr])

            # Rerank with timeout simulation (using simple computation)
            scores = self._model.predict(pairs)

            # Match back to candidates
            for idx, score in enumerate(scores):
                candidates[idx]["rerank_score"] = float(score)

            # Sort by rerank score descending
            reranked = sorted(
                candidates, key=lambda x: x.get("rerank_score", -9999.0), reverse=True
            )
            logger.info(
                f"Reranking of {len(candidates)} candidates completed in {time.perf_counter() - start_time:.4f}s."
            )
            return reranked[:top_k]

        except Exception as e:
            logger.error(f"Reranker prediction failed: {e}")
            raise SearchError(
                code=RERANKER_FAILED,
                message=f"Reranking failed: {e}",
                subsystem="RERANKER",
                retry_safe=True,
            )
