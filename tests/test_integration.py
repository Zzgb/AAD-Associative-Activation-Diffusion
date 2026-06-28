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
    def test_lookup_huangrenxun(self, seeded_system):
        store, _ = seeded_system
        result = aad_lookup(store, "黄仁勋")
        assert result["ok"] is True
        node = result["node"]
        assert node["name"] == "黄仁勋"
        assert len(node["associations"]) == 2
        # all associations use ref
        for a in node["associations"]:
            assert "ref" in a
            assert "reason" in a
            assert "vector" not in a

    def test_expand_from_ref_finds_nvidia(self, seeded_system):
        store, index = seeded_system
        lookup = aad_lookup(store, "黄仁勋")
        # Find the NVIDIA association ref
        nvidia_assoc = next(
            a for a in lookup["node"]["associations"]
            if "NVIDIA" in a["reason"]
        )
        result = aad_expand(store, index, ref=nvidia_assoc["ref"], top_k=3)
        assert result["ok"] is True
        names = {r["name"] for r in result["results"]}
        assert "NVIDIA" in names

    def test_get_nvidia_content(self, seeded_system):
        store, _ = seeded_system
        result = aad_get_content(store, "NVIDIA")
        assert result["ok"] is True
        assert "黄仁勋" in result["content"]

    def test_full_reasoning_chain(self, seeded_system):
        """Complete: 黄仁勋 → NVIDIA → GPU traversal."""
        store, index = seeded_system

        # Lookup 黄仁勋
        r1 = aad_lookup(store, "黄仁勋")
        assert r1["ok"]
        nvidia_ref = next(
            a["ref"] for a in r1["node"]["associations"]
            if "NVIDIA" in a["reason"]
        )

        # Expand to NVIDIA
        r2 = aad_expand(store, index, ref=nvidia_ref, top_k=3)
        assert r2["ok"]
        nvidia = next(r for r in r2["results"] if r["name"] == "NVIDIA")
        gpu_ref = next(
            a["ref"] for a in nvidia["associations"]
            if "GPU" in a["reason"]
        )

        # Expand to GPU
        r3 = aad_expand(store, index, ref=gpu_ref, top_k=3)
        assert r3["ok"]
        assert any(r["name"] == "GPU" for r in r3["results"])

        # Get GPU content
        r4 = aad_get_content(store, "GPU")
        assert r4["ok"]
        assert "图形处理单元" in r4["content"]

    def test_invalid_ref_returns_error(self, seeded_system):
        store, index = seeded_system
        result = aad_expand(store, index, ref="nonexistent", top_k=3)
        assert result["ok"] is False

    def test_not_found_returns_error_with_available_nodes(self, seeded_system):
        store, _ = seeded_system
        result = aad_lookup(store, "QuantumComputing")
        assert result["ok"] is False
        assert "已知节点" in result["error"]

    def test_all_nodes_reachable(self, seeded_system):
        store, _ = seeded_system
        for name in ["黄仁勋", "GPU", "NVIDIA"]:
            lookup = aad_lookup(store, name)
            assert lookup["ok"], f"aad_lookup({name!r}) failed"
            content = aad_get_content(store, name)
            assert content["ok"], f"aad_get_content({name!r}) failed"
            assert len(content["content"]) > 0
