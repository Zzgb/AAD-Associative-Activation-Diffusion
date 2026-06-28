"""End-to-end integration tests: seed data + tool chain with session memory."""

import pytest
import tempfile
import hashlib
from pathlib import Path
from unittest.mock import MagicMock

from aad.store import AADStore
from aad.vector_index import VectorIndex
from aad.session import SessionMemory
from aad.seed import create_seed_nodes
from aad.tools import aad_lookup, aad_expand, aad_get_content, _clear_refs

DIM = 12


def _fake_embedder(dim=DIM):
    def fake_embed(text):
        h = hashlib.sha256(text.encode()).digest()[:dim]
        return [float(b) / 255.0 for b in h]
    emb = MagicMock()
    emb.embed.side_effect = fake_embed
    emb.embed_batch.side_effect = lambda texts: [fake_embed(t) for t in texts]
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

        session = SessionMemory(_fake_embedder(DIM), dim=DIM)
        _clear_refs()
        yield store, index, session


class TestIntegration:
    def test_lookup_mirrors_to_session(self, seeded_system):
        store, _, session = seeded_system
        result = aad_lookup(store, session, "黄仁勋")
        assert result["ok"]
        assert result["source"] == "long_term"
        # Mirrored
        assert session.is_visited("黄仁勋")
        # Second call from session
        result2 = aad_lookup(store, session, "黄仁勋")
        assert result2["source"] == "short_term"

    def test_expand_finds_nvidia(self, seeded_system):
        store, index, session = seeded_system
        lookup = aad_lookup(store, session, "黄仁勋")
        nvidia_ref = next(a["ref"] for a in lookup["node"]["associations"] if "NVIDIA" in a["reason"])
        result = aad_expand(store, index, session, ref=nvidia_ref, top_k=3)
        assert result["ok"]
        names = {r["name"] for r in result["results"]}
        assert "NVIDIA" in names

    def test_full_chain_to_deeplearning(self, seeded_system):
        store, index, session = seeded_system

        r1 = aad_lookup(store, session, "黄仁勋")
        nvidia_ref = next(a["ref"] for a in r1["node"]["associations"] if "NVIDIA" in a["reason"])

        r2 = aad_expand(store, index, session, ref=nvidia_ref, top_k=3)
        nvidia = next(r for r in r2["results"] if r["name"] == "NVIDIA")
        cuda_ref = next(a["ref"] for a in nvidia["associations"] if "CUDA" in a["reason"])

        r3 = aad_expand(store, index, session, ref=cuda_ref, top_k=3)
        cuda = next(r for r in r3["results"] if r["name"] == "CUDA")
        dl_ref = next(a["ref"] for a in cuda["associations"] if "深度学习" in a["reason"])

        r4 = aad_expand(store, index, session, ref=dl_ref, top_k=3)
        assert any(r["name"] == "深度学习" for r in r4["results"])

    def test_distractors_no_associations(self, seeded_system):
        store, _, session = seeded_system
        for name in ["苹果", "特斯拉", "Python"]:
            result = aad_lookup(store, session, name)
            assert result["ok"]
            assert result["node"]["associations"] == []

    def test_session_accumulates_mirrors(self, seeded_system):
        store, _, session = seeded_system
        assert session.mirrored_count == 0
        aad_lookup(store, session, "GPU")
        aad_lookup(store, session, "NVIDIA")
        aad_lookup(store, session, "黄仁勋")
        assert session.mirrored_count == 3

    def test_not_found_shows_known_nodes(self, seeded_system):
        store, _, session = seeded_system
        result = aad_lookup(store, session, "不存在")
        assert result["ok"] is False
        assert "已知节点" in result["error"]

    def test_invalid_ref_error(self, seeded_system):
        store, index, session = seeded_system
        result = aad_expand(store, index, session, ref="bad", top_k=3)
        assert result["ok"] is False
