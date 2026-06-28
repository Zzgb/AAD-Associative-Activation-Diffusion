"""FAISS vector index wrapper for AAD node embeddings."""

import numpy as np
import faiss

from aad.errors import VectorIndexError


class VectorIndex:
    """FAISS index for ANN search over node embedding vectors.

    Uses IndexFlatIP (inner product) because we assume normalized vectors,
    making inner product equivalent to cosine similarity.

    Maintains a mapping from FAISS internal IDs to node names so that
    search results return node names, not raw IDs.
    """

    def __init__(self, dim: int) -> None:
        if dim <= 0:
            raise VectorIndexError(f"Vector dimension must be positive, got {dim}")
        self._dim = dim
        self._index = faiss.IndexFlatIP(dim)
        self._id_to_name: dict[int, str] = {}
        self._name_to_id: dict[str, int] = {}
        self._next_id: int = 0

    def add(self, name: str, vector: list[float]) -> None:
        """Add or update a vector in the index, keyed by node name."""
        arr = self._validate_vector(vector)
        if name in self._name_to_id:
            self.remove(name)
        faiss_id = self._next_id
        self._index.add(arr)
        self._id_to_name[faiss_id] = name
        self._name_to_id[name] = faiss_id
        self._next_id += 1

    def search(
        self, query_vector: list[float], top_k: int = 3
    ) -> list[tuple[str, float]]:
        """Search for the top_k nearest neighbors. Returns list of (name, score)."""
        if self._index.ntotal == 0:
            return []
        arr = self._validate_vector(query_vector)
        distances, indices = self._index.search(arr, min(top_k, self._index.ntotal))
        results: list[tuple[str, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0 and idx in self._id_to_name:
                results.append((self._id_to_name[int(idx)], float(dist)))
        return results

    def remove(self, name: str) -> None:
        """Remove a vector from the index by node name.

        Note: FAISS IndexFlatIP does not support efficient removal.
        We mark the entry as removed in our mappings and require
        a rebuild to reclaim space. For small datasets this is acceptable.
        """
        if name in self._name_to_id:
            faiss_id = self._name_to_id[name]
            self._id_to_name.pop(faiss_id, None)
            self._name_to_id.pop(name, None)

    def rebuild(self, nodes: dict[str, "Node"]) -> None:
        """Rebuild the index from a dict of name -> Node."""
        from aad.models import Node

        self._index = faiss.IndexFlatIP(self._dim)
        self._id_to_name.clear()
        self._name_to_id.clear()
        self._next_id = 0
        for name, node in nodes.items():
            if node.vector:
                self.add(name, node.vector)

    def __len__(self) -> int:
        return len(self._name_to_id)

    def _validate_vector(self, vector: list[float]) -> np.ndarray:
        """Convert to numpy array and validate dimensions."""
        arr = np.array(vector, dtype=np.float32).reshape(1, -1)
        if arr.shape[1] != self._dim:
            raise VectorIndexError(
                f"Vector dimension mismatch: expected {self._dim}, got {arr.shape[1]}"
            )
        return arr
