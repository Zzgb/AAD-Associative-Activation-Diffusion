"""Embedding generation via simple deterministic hashing.

For prototype validation only — generates fixed-dimension vectors from
text via SHA-256, no external API or model download needed.
Replace with real embedding API (OpenAI / DeepSeek) in production.
"""

import hashlib


class Embedder:
    """Deterministic hash-based embedder for prototype validation.

    Usage:
        embedder = Embedder(dim=256)
        vec = embedder.embed("some text")
        vecs = embedder.embed_batch(["text1", "text2"])
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0 or dim > 1024:
            raise ValueError(f"dim must be 1–1024, got {dim}")
        self._dim = dim

    def embed(self, text: str) -> list[float]:
        """Generate a deterministic vector from text hash."""
        if not text.strip():
            return [0.0] * self._dim
        return self._hash_to_vector(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate deterministic vectors for multiple texts."""
        if not texts:
            return []
        return [self.embed(t) for t in texts]

    def _hash_to_vector(self, text: str) -> list[float]:
        """SHA-256 → normalized float vector of configured dimension."""
        h = hashlib.sha256(text.encode()).digest()
        # Expand hash to fill dim floats by cycling
        vec = []
        for i in range(self._dim):
            b = h[i % len(h)]
            # Normalize to [0, 1]
            vec.append(b / 255.0)
        # Normalize to unit vector (for FAISS IndexFlatIP = cosine similarity)
        norm = sum(v * v for v in vec) ** 0.5
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec
