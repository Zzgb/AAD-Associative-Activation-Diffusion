"""Embedding generation using local sentence-transformers model.

No external API calls needed — runs entirely locally.
Default model: BAAI/bge-small-zh-v1.5 (512-d, optimized for Chinese).
"""

from sentence_transformers import SentenceTransformer

from aad.errors import EmbeddingError


class Embedder:
    """Generates embeddings via a local sentence-transformers model.

    Usage:
        embedder = Embedder(model_name="BAAI/bge-small-zh-v1.5")
        vec = embedder.embed("some text")
        vecs = embedder.embed_batch(["text1", "text2"])
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
    ) -> None:
        if not model_name:
            raise EmbeddingError("Embedding model name is not set")
        try:
            self._model = SentenceTransformer(model_name)
        except Exception as exc:
            raise EmbeddingError(f"Failed to load model {model_name!r}: {exc}") from exc

    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text string."""
        if not text.strip():
            raise EmbeddingError("Cannot embed empty text")
        try:
            result = self._model.encode(text, normalize_embeddings=True)
            return result.tolist()
        except Exception as exc:
            raise EmbeddingError(str(exc)) from exc

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for multiple texts."""
        if not texts:
            return []
        non_empty = [t for t in texts if t.strip()]
        if not non_empty:
            return [[] for _ in texts]
        try:
            results = self._model.encode(
                non_empty, normalize_embeddings=True, show_progress_bar=False
            )
            return [r.tolist() for r in results]
        except Exception as exc:
            raise EmbeddingError(str(exc)) from exc
