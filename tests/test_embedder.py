import pytest
from unittest.mock import patch, MagicMock
from aad.embedder import Embedder
from aad.errors import EmbeddingError


class TestEmbedder:
    def test_init_requires_api_key(self):
        with pytest.raises(EmbeddingError, match="DEEPSEEK_API_KEY"):
            Embedder(api_key="")

    def test_init_with_key(self):
        e = Embedder(api_key="sk-test")
        assert e._model == "deepseek-chat"

    def test_embed_returns_vector(self):
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]

        with patch.object(Embedder, "__init__", lambda self: None):
            e = Embedder.__new__(Embedder)
            e._model = "deepseek-chat"
            e._client = MagicMock()
            e._client.embeddings.create.return_value = mock_response

            result = e.embed("hello world")
            assert result == [0.1, 0.2, 0.3]
            e._client.embeddings.create.assert_called_once_with(
                model="deepseek-chat",
                input="hello world",
            )

    def test_embed_empty_text_raises(self):
        with patch.object(Embedder, "__init__", lambda self: None):
            e = Embedder.__new__(Embedder)
            e._model = "deepseek-chat"
            e._client = MagicMock()
            with pytest.raises(EmbeddingError, match="empty"):
                e.embed("   ")

    def test_embed_api_error_wraps(self):
        with patch.object(Embedder, "__init__", lambda self: None):
            e = Embedder.__new__(Embedder)
            e._model = "deepseek-chat"
            e._client = MagicMock()
            e._client.embeddings.create.side_effect = RuntimeError("timeout")

            with pytest.raises(EmbeddingError, match="timeout"):
                e.embed("hello")

    def test_embed_batch_multiple_texts(self):
        mock_response = MagicMock()
        mock_response.data = [
            MagicMock(embedding=[1.0, 2.0]),
            MagicMock(embedding=[3.0, 4.0]),
        ]
        with patch.object(Embedder, "__init__", lambda self: None):
            e = Embedder.__new__(Embedder)
            e._model = "deepseek-chat"
            e._client = MagicMock()
            e._client.embeddings.create.return_value = mock_response

            result = e.embed_batch(["text1", "text2"])
            assert result == [[1.0, 2.0], [3.0, 4.0]]

    def test_embed_batch_empty_list(self):
        with patch.object(Embedder, "__init__", lambda self: None):
            e = Embedder.__new__(Embedder)
            e._model = "deepseek-chat"
            e._client = MagicMock()
            result = e.embed_batch([])
            assert result == []
