import pytest
import hashlib
from unittest.mock import MagicMock

from aad.seed import create_seed_nodes


def _fake_embedder():
    """Mock embedder that returns deterministic vectors by hashing text."""
    def fake_embed(text):
        h = hashlib.sha256(text.encode()).digest()[:12]
        return [float(b) / 255.0 for b in h]

    emb = MagicMock()
    emb.embed_batch.side_effect = lambda texts: [fake_embed(t) for t in texts]
    emb.embed.side_effect = fake_embed
    return emb


class TestSeedData:
    @pytest.fixture
    def mock_embedder(self):
        return _fake_embedder()

    def test_creates_three_nodes(self, mock_embedder):
        nodes = create_seed_nodes(mock_embedder)
        assert len(nodes) == 3

    def test_all_nodes_have_vectors(self, mock_embedder):
        nodes = create_seed_nodes(mock_embedder)
        for node in nodes:
            assert len(node.vector) == 12
            assert all(isinstance(v, float) for v in node.vector)

    def test_nodes_have_expected_names(self, mock_embedder):
        nodes = create_seed_nodes(mock_embedder)
        names = {n.name for n in nodes}
        assert names == {"黄仁勋", "GPU", "NVIDIA"}

    def test_huangrenxun_has_two_associations(self, mock_embedder):
        nodes = create_seed_nodes(mock_embedder)
        jensen = next(n for n in nodes if n.name == "黄仁勋")
        assert len(jensen.associations) == 2
        reasons = {a.reason for a in jensen.associations}
        assert "联合创立了 NVIDIA（1993 年）" in reasons
        assert "领导了发明 GPU 的公司" in reasons

    def test_gpu_has_two_associations(self, mock_embedder):
        nodes = create_seed_nodes(mock_embedder)
        gpu = next(n for n in nodes if n.name == "GPU")
        assert len(gpu.associations) == 2
        reasons = {a.reason for a in gpu.associations}
        assert "GPU 由 NVIDIA 发明" in reasons

    def test_nvidia_has_two_associations(self, mock_embedder):
        nodes = create_seed_nodes(mock_embedder)
        nvidia = next(n for n in nodes if n.name == "NVIDIA")
        assert len(nvidia.associations) == 2

    def test_associations_are_bidirectional(self, mock_embedder):
        """Verify the graph is fully connected (every node links to both others)."""
        nodes = create_seed_nodes(mock_embedder)
        for source in nodes:
            target_names_in_associations = set()
            assert len(source.associations) == 2
            for assoc in source.associations:
                for target in nodes:
                    if target.name != source.name and assoc.vector == target.vector:
                        target_names_in_associations.add(target.name)
            assert len(target_names_in_associations) == 2, (
                f"{source.name} should associate to both other nodes, "
                f"got {target_names_in_associations}"
            )

    def test_nodes_have_meaningful_content(self, mock_embedder):
        nodes = create_seed_nodes(mock_embedder)
        jensen = next(n for n in nodes if n.name == "黄仁勋")
        assert "NVIDIA" in jensen.content
        assert "1993" in jensen.content

        gpu = next(n for n in nodes if n.name == "GPU")
        assert "图形处理单元" in gpu.content

        nvidia = next(n for n in nodes if n.name == "NVIDIA")
        assert "黄仁勋" in nvidia.content

    def test_embedding_generated_from_name_plus_content(self, mock_embedder):
        """Verify embed_batch is called with name + content concatenation."""
        nodes = create_seed_nodes(mock_embedder)
        # Check embed_batch was called with name + " " + content
        call_args = mock_embedder.embed_batch.call_args[0][0]
        assert len(call_args) == 3
        for text in call_args:
            assert any(name in text for name in ["黄仁勋", "GPU", "NVIDIA"])
