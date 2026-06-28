import pytest
from aad.embedder import Embedder


class TestEmbedder:
    def test_init_dim_validation(self):
        with pytest.raises(ValueError):
            Embedder(dim=0)
        with pytest.raises(ValueError):
            Embedder(dim=-1)
        with pytest.raises(ValueError):
            Embedder(dim=2000)

    def test_init_default_dim(self):
        e = Embedder()
        assert e._dim == 256

    def test_embed_returns_list_of_floats(self):
        e = Embedder(dim=16)
        result = e.embed("hello world")
        assert len(result) == 16
        assert all(isinstance(v, float) for v in result)

    def test_embed_deterministic(self):
        e = Embedder(dim=16)
        r1 = e.embed("hello")
        r2 = e.embed("hello")
        assert r1 == r2

    def test_embed_different_texts_different_vectors(self):
        e = Embedder(dim=16)
        r1 = e.embed("hello")
        r2 = e.embed("world")
        assert r1 != r2

    def test_embed_empty_text_returns_zeros(self):
        e = Embedder(dim=16)
        result = e.embed("   ")
        assert result == [0.0] * 16

    def test_embed_unit_vector(self):
        """Vector should be normalized (unit length)."""
        e = Embedder(dim=64)
        result = e.embed("test")
        norm = sum(v * v for v in result) ** 0.5
        assert norm == pytest.approx(1.0)

    def test_embed_batch_multiple_texts(self):
        e = Embedder(dim=16)
        result = e.embed_batch(["text1", "text2"])
        assert len(result) == 2
        assert len(result[0]) == 16
        assert len(result[1]) == 16

    def test_embed_batch_empty_list(self):
        e = Embedder(dim=16)
        result = e.embed_batch([])
        assert result == []
