"""Rank fusion for the hybrid retrieval pipeline."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

__all__ = ["reciprocal_rank_fusion"]


def reciprocal_rank_fusion(
    channels: Iterable[Tuple[Sequence[str], float]],
    k: float = 60.0,
) -> Dict[str, float]:
    """Fuses ranked id lists into one score map.

    Each channel contributes ``weight / (k + rank)`` per document, with ``rank``
    being 1-based. RRF needs no score calibration between channels, which is
    what makes a lexical BM25 rank and a vector distance rank combinable.
    """
    scores: Dict[str, float] = {}
    for ranked_ids, weight in channels:
        for index, identifier in enumerate(ranked_ids):
            scores[identifier] = scores.get(identifier, 0.0) + weight / (k + index + 1)
    return scores


def top_n(scores: Dict[str, float], limit: int) -> List[Tuple[str, float]]:
    """Returns the ``limit`` highest-scoring ``(id, score)`` pairs."""
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
