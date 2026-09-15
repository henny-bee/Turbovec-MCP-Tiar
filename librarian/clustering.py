"""DBSCAN over embedding vectors.

The pairwise distance matrix is computed with the ``|a-b|² = |a|² + |b|² - 2ab``
identity, which is one BLAS matrix product instead of an ``O(n²)`` Python loop
and keeps memory at ``n²`` floats rather than ``n² * dim``.
"""

from __future__ import annotations

from typing import List

import numpy as np

__all__ = ["DBSCANClustering", "pairwise_distances"]


def pairwise_distances(embeddings: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix for the rows of ``embeddings``."""
    squared_norms = np.einsum("ij,ij->i", embeddings, embeddings)
    squared = (
        squared_norms[:, None]
        + squared_norms[None, :]
        - 2.0 * (embeddings @ embeddings.T)
    )
    np.maximum(squared, 0.0, out=squared)
    distances = np.sqrt(squared)
    np.fill_diagonal(distances, 0.0)
    return distances


class DBSCANClustering:
    def __init__(self, eps: float = 0.5, min_samples: int = 2) -> None:
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, embeddings: np.ndarray) -> List[int]:
        """Returns one cluster label per row; ``-1`` marks noise."""
        sample_count = len(embeddings)
        if sample_count == 0:
            return []

        matrix = np.asarray(embeddings, dtype=np.float32)
        distances = pairwise_distances(matrix)
        labels = [-1] * sample_count
        visited: set = set()

        def neighbors_of(index: int) -> List[int]:
            return np.nonzero(distances[index] <= self.eps)[0].tolist()

        cluster_id = 0
        for seed in range(sample_count):
            if seed in visited:
                continue
            visited.add(seed)

            neighbors = neighbors_of(seed)
            if len(neighbors) < self.min_samples:
                labels[seed] = -1
                continue

            # Core point: expand the cluster breadth-first over density-reachable
            # points, absorbing border points as it goes.
            labels[seed] = cluster_id
            queue = [index for index in neighbors if index != seed]
            queued = set(queue)

            position = 0
            while position < len(queue):
                candidate = queue[position]
                if candidate not in visited:
                    visited.add(candidate)
                    candidate_neighbors = neighbors_of(candidate)
                    if len(candidate_neighbors) >= self.min_samples:
                        for index in candidate_neighbors:
                            if index not in queued and index not in visited:
                                queue.append(index)
                                queued.add(index)
                if labels[candidate] == -1:
                    labels[candidate] = cluster_id
                position += 1

            cluster_id += 1

        return labels
