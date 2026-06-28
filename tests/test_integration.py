"""End-to-end integration tests: seed data + tool chain without LLM."""

import pytest
import tempfile
import hashlib
from pathlib import Path
from unittest.mock import MagicMock

from aad.store import AADStore
from aad.vector_index import VectorIndex
from aad.seed import create_seed_nodes
from aad.tools import aad_lookup, aad_expand, aad_get_content, _clear_refs

DIM = 12


def _fake_embedder():
    def fake_embed(text):
        h = hashlib.sha256(text.encode()).digest()[:12]
        return [float(b) / 255.0 for b in h]

    emb = MagicMock()
    emb.embed_batch.side_effect = lambda texts: [fake_embed(t) for t in texts]
    emb.embed.side_effect = fake_embed
    return emb


@pytest.fixture(autouse=True)
def clear_refs():
    _clear_refs()
    yield


@pytest.fixture
def seeded_system():
    embedder = _fake_embedder()
    nodes = create_seed_nodes(embedder)

    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = Path(tmpdir) / "test.jsonl"
        store = AADStore(str(store_path))
        for node in nodes:
            store.put(node)

        index = VectorIndex(dim=DIM)
        index.rebuild(store._nodes)

        _clear_refs()
        yield store, index


class TestIntegration:
    def test_all_core_nodes_reachable(self, seeded_system):
        """Every core node can be looked up."""
        store, _ = seeded_system
        for name in ["黄仁勋", "NVIDIA", "GPU", "CUDA", "深度学习", "AMD"]:
            result = aad_lookup(store, name)
            assert result["ok"], f"aad_lookup({name!r}) failed"

    def test_distractors_have_no_associations(self, seeded_system):
        """Distractor nodes exist but have no links."""
        store, _ = seeded_system
        for name in ["苹果", "特斯拉", "Python"]:
            result = aad_lookup(store, name)
            assert result["ok"]
            assert result["node"]["associations"] == []

    def test_lookup_returns_ref_based_associations(self, seeded_system):
        """Associations contain ref (string), not raw vector."""
        store, _ = seeded_system
        result = aad_lookup(store, "NVIDIA")
        assert result["ok"]
        for a in result["node"]["associations"]:
            assert isinstance(a["ref"], str)
            assert "reason" in a
            assert "vector" not in a

    def test_expand_from_huangrenxun_to_nvidia(self, seeded_system):
        """Expand from 黄仁勋's NVIDIA association finds NVIDIA."""
        store, index = seeded_system
        lookup = aad_lookup(store, "黄仁勋")
        nvidia_ref = next(
            a["ref"] for a in lookup["node"]["associations"]
            if "联合创立了 NVIDIA" in a["reason"]
        )
        result = aad_expand(store, index, ref=nvidia_ref, top_k=3)
        assert result["ok"]
        names = {r["name"] for r in result["results"]}
        assert "NVIDIA" in names

    def test_full_chain_huangrenxun_to_deeplearning(self, seeded_system):
        """Traversal: 黄仁勋 → NVIDIA → CUDA → 深度学习."""
        store, index = seeded_system

        r1 = aad_lookup(store, "黄仁勋")
        nvidia_ref = next(a["ref"] for a in r1["node"]["associations"] if "NVIDIA" in a["reason"])

        r2 = aad_expand(store, index, ref=nvidia_ref, top_k=3)
        nvidia = next(r for r in r2["results"] if r["name"] == "NVIDIA")
        cuda_ref = next(a["ref"] for a in nvidia["associations"] if "CUDA" in a["reason"])

        r3 = aad_expand(store, index, ref=cuda_ref, top_k=3)
        assert r3["ok"]
        cuda_results = {r["name"] for r in r3["results"]}
        assert "CUDA" in cuda_results

        cuda = next(r for r in r3["results"] if r["name"] == "CUDA")
        dl_ref = next(a["ref"] for a in cuda["associations"] if "深度学习" in a["reason"])

        r4 = aad_expand(store, index, ref=dl_ref, top_k=3)
        dl_names = {r["name"] for r in r4["results"]}
        assert "深度学习" in dl_names

    def test_gpu_expand_finds_related(self, seeded_system):
        """Expanding from GPU finds multiple related concepts."""
        store, index = seeded_system
        gpu = aad_lookup(store, "GPU")
        # Pick a ref and expand
        ge_ref = next(a["ref"] for a in gpu["node"]["associations"] if "图形渲染" in a["reason"])
        result = aad_expand(store, index, ref=ge_ref, top_k=5)
        assert result["ok"]
        names = {r["name"] for r in result["results"]}
        assert "图形渲染" in names

    def test_not_found_shows_available_nodes(self, seeded_system):
        store, _ = seeded_system
        result = aad_lookup(store, "不存在")
        assert result["ok"] is False
        assert "已知节点" in result["error"]

    def test_invalid_ref_returns_error(self, seeded_system):
        store, index = seeded_system
        result = aad_expand(store, index, ref="bad", top_k=3)
        assert result["ok"] is False
