"""Embedding generation via DeepSeek's OpenAI-compatible API."""

from openai import OpenAI

from aad.errors import EmbeddingError


class Embedder:
    """Generates embeddings via DeepSeek's embedding API.

    Usage:
        embedder = Embedder(api_key="sk-...", model="deepseek-chat")
        vec = embedder.embed("some text")
        vecs = embedder.embed_batch(["text1", "text2"])
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
    ) -> None:
        if not api_key:
            raise EmbeddingError("DEEPSEEK_API_KEY is not set")
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text string."""
        if not text.strip():
            raise EmbeddingError("Cannot embed empty text")
        try:
            response = self._client.embeddings.create(
                model=self._model,
                input=text,
            )
            return response.data[0].embedding
        except Exception as exc:
            raise EmbeddingError(str(exc)) from exc

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for multiple texts in one API call."""
        if not texts:
            return []
        non_empty = [t for t in texts if t.strip()]
        if not non_empty:
            return [[] for _ in texts]
        try:
            response = self._client.embeddings.create(
                model=self._model,
                input=non_empty,
            )
            return [d.embedding for d in response.data]
        except Exception as exc:
            raise EmbeddingError(str(exc)) from exc
