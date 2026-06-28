import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from aad.embedder import Embedder
from aad.errors import EmbeddingError


class TestEmbedder:
    def test_init_requires_model_name(self):
        with pytest.raises(EmbeddingError, match="not set"):
            Embedder(model_name="")

    def test_init_loads_model(self):
        """Verify the model name is stored."""
        with patch("aad.embedder.SentenceTransformer") as mock_st:
            mock_st.return_value = MagicMock()
            e = Embedder(model_name="test-model")
            assert e._model is mock_st.return_value
            mock_st.assert_called_once_with("test-model")

    def test_init_model_load_failure_wraps(self):
        with patch("aad.embedder.SentenceTransformer") as mock_st:
            mock_st.side_effect = RuntimeError("download failed")
            with pytest.raises(EmbeddingError, match="Failed to load model"):
                Embedder(model_name="bad-model")

    def test_embed_returns_list_of_floats(self):
        with patch("aad.embedder.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.array([0.1, 0.2, 0.3], dtype=np.float32)
            mock_st.return_value = mock_model

            e = Embedder(model_name="test-model")
            result = e.embed("hello world")
            assert result == pytest.approx([0.1, 0.2, 0.3])
            mock_model.encode.assert_called_once_with(
                "hello world", normalize_embeddings=True
            )

    def test_embed_empty_text_raises(self):
        with patch("aad.embedder.SentenceTransformer") as mock_st:
            mock_st.return_value = MagicMock()
            e = Embedder(model_name="test-model")
            with pytest.raises(EmbeddingError, match="empty"):
                e.embed("   ")

    def test_embed_encode_error_wraps(self):
        with patch("aad.embedder.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.encode.side_effect = RuntimeError("OOM")
            mock_st.return_value = mock_model

            e = Embedder(model_name="test-model")
            with pytest.raises(EmbeddingError, match="OOM"):
                e.embed("hello")

    def test_embed_batch_multiple_texts(self):
        with patch("aad.embedder.SentenceTransformer") as mock_st:
            mock_model = MagicMock()
            mock_model.encode.return_value = np.array(
                [[1.0, 2.0], [3.0, 4.0]], dtype=np.float32
            )
            mock_st.return_value = mock_model

            e = Embedder(model_name="test-model")
            result = e.embed_batch(["text1", "text2"])
            assert len(result) == 2
            mock_model.encode.assert_called_once_with(
                ["text1", "text2"], normalize_embeddings=True, show_progress_bar=False
            )

    def test_embed_batch_empty_list(self):
        with patch("aad.embedder.SentenceTransformer") as mock_st:
            mock_st.return_value = MagicMock()
            e = Embedder(model_name="test-model")
            result = e.embed_batch([])
            assert result == []
