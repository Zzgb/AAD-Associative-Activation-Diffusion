import pytest
import hashlib
from unittest.mock import MagicMock

from aad.seed import create_seed_nodes


def _fake_embedder():
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

    def test_creates_correct_number_of_nodes(self, mock_embedder):
        nodes = create_seed_nodes(mock_embedder)
        assert len(nodes) == 17

    def test_all_nodes_have_vectors(self, mock_embedder):
        nodes = create_seed_nodes(mock_embedder)
        for node in nodes:
            assert len(node.vector) == 12
            assert all(isinstance(v, float) for v in node.vector)

    def test_core_nodes_exist(self, mock_embedder):
        nodes = create_seed_nodes(mock_embedder)
        names = {n.name for n in nodes}
        assert "黄仁勋" in names
        assert "NVIDIA" in names
        assert "GPU" in names

    def test_gpu_related_nodes_exist(self, mock_embedder):
        nodes = create_seed_nodes(mock_embedder)
        names = {n.name for n in nodes}
        for expected in ["深度学习", "CUDA", "图形渲染", "GeForce", "RTX", "数据中心"]:
            assert expected in names, f"Missing: {expected}"

    def test_distractor_nodes_exist(self, mock_embedder):
        nodes = create_seed_nodes(mock_embedder)
        names = {n.name for n in nodes}
        for expected in ["苹果", "特斯拉", "Python", "量子计算"]:
            assert expected in names, f"Missing: {expected}"

    def test_distractors_have_no_associations(self, mock_embedder):
        """Distractor nodes should have zero associations."""
        nodes = create_seed_nodes(mock_embedder)
        for name in ["苹果", "特斯拉", "Python", "量子计算"]:
            node = next(n for n in nodes if n.name == name)
            assert len(node.associations) == 0, (
                f"{name} should have 0 associations, got {len(node.associations)}"
            )

    def test_huangrenxun_has_multiple_associations(self, mock_embedder):
        nodes = create_seed_nodes(mock_embedder)
        jensen = next(n for n in nodes if n.name == "黄仁勋")
        # Should have: NVIDIA, GPU, CUDA, AMD = 4
        assert len(jensen.associations) >= 3

    def test_nvidia_has_many_associations(self, mock_embedder):
        nodes = create_seed_nodes(mock_embedder)
        nvidia = next(n for n in nodes if n.name == "NVIDIA")
        # 黄仁勋, GPU, CUDA, GeForce, 数据中心, RTX, AMD = 7
        assert len(nvidia.associations) >= 5

    def test_associations_use_target_vectors(self, mock_embedder):
        """Every association's vector should match the target node's vector."""
        nodes = create_seed_nodes(mock_embedder)
        name_to_vec = {n.name: n.vector for n in nodes}
        for node in nodes:
            for assoc in node.associations:
                # Find which target node has this vector
                matched = [
                    name for name, vec in name_to_vec.items()
                    if vec == assoc.vector and name != node.name
                ]
                assert len(matched) == 1, (
                    f"{node.name}'s association '{assoc.reason}' "
                    f"points to vector of: {matched if matched else 'NO ONE'}"
                )

    def test_nvidia_gpu_reason_includes_key_words(self, mock_embedder):
        nodes = create_seed_nodes(mock_embedder)
        nvidia = next(n for n in nodes if n.name == "NVIDIA")
        reasons = {a.reason for a in nvidia.associations}
        assert any("GPU" in r for r in reasons)
        assert any("CUDA" in r for r in reasons)
        assert any("AMD" in r for r in reasons)

    def test_cuda_links_to_deeplearning(self, mock_embedder):
        nodes = create_seed_nodes(mock_embedder)
        cuda = next(n for n in nodes if n.name == "CUDA")
        reasons = {a.reason for a in cuda.associations}
        assert any("深度学习" in r for r in reasons)
