import numpy as np
from typing import List, Dict, Set, Any

class DBSCANClustering:
    def __init__(self, eps: float = 0.5, min_samples: int = 2):
        self.eps = eps
        self.min_samples = min_samples

    def fit(self, embeddings: np.ndarray) -> List[int]:
        """
        Fits DBSCAN on embeddings.
        Returns a list of cluster labels where -1 represents noise points.
        """
        n_samples = len(embeddings)
        labels = [-1] * n_samples
        visited = set()

        # Compute distance matrix (Euclidean distance)
        # To handle empty or small lists of embeddings gracefully
        if n_samples == 0:
            return []

        dist_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i, n_samples):
                dist = float(np.linalg.norm(embeddings[i] - embeddings[j]))
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        def get_neighbors(index: int) -> List[int]:
            return [idx for idx, dist in enumerate(dist_matrix[index]) if dist <= self.eps]

        cluster_id = 0
        for i in range(n_samples):
            if i in visited:
                continue
            visited.add(i)

            neighbors = get_neighbors(i)
            if len(neighbors) < self.min_samples:
                labels[i] = -1
            else:
                # Expand cluster
                labels[i] = cluster_id
                queue = list(neighbors)
                queue.remove(i) if i in queue else None
                
                # Use while loop to expand
                idx_in_queue = 0
                while idx_in_queue < len(queue):
                    neighbor_idx = queue[idx_in_queue]
                    if neighbor_idx not in visited:
                        visited.add(neighbor_idx)
                        neigh_neighbors = get_neighbors(neighbor_idx)
                        if len(neigh_neighbors) >= self.min_samples:
                            # Add unvisited neighbors to queue
                            for n in neigh_neighbors:
                                if n not in queue and n not in visited:
                                    queue.append(n)
                    
                    if labels[neighbor_idx] == -1:
                        labels[neighbor_idx] = cluster_id
                    
                    idx_in_queue += 1

                cluster_id += 1

        return labels
