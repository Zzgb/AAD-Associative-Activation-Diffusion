"""End-to-end integration tests: seed data + tool chain without LLM.

These tests verify that the seed data graph is traversable using the
three tools in the pattern an agent would use.
"""

import pytest
import tempfile
import hashlib
from pathlib import Path
from unittest.mock import MagicMock

from aad.store import AADStore
from aad.vector_index import VectorIndex
from aad.seed import create_seed_nodes
from aad.tools import aad_lookup, aad_expand, aad_get_content

DIM = 12  # matches mock embedder output


def _fake_embedder():
    """Return a mock embedder that produces deterministic 12-d vectors."""
    def fake_embed(text):
        h = hashlib.sha256(text.encode()).digest()[:12]
        return [float(b) / 255.0 for b in h]

    emb = MagicMock()
    emb.embed_batch.side_effect = lambda texts: [fake_embed(t) for t in texts]
    emb.embed.side_effect = fake_embed
    return emb


@pytest.fixture
def seeded_system():
    """Create a fully seeded store + index."""
    embedder = _fake_embedder()
    nodes = create_seed_nodes(embedder)

    with tempfile.TemporaryDirectory() as tmpdir:
        store_path = Path(tmpdir) / "test.jsonl"
        store = AADStore(str(store_path))
        for node in nodes:
            store.put(node)

        index = VectorIndex(dim=DIM)
        index.rebuild(store._nodes)

        yield store, index


class TestIntegration:
    """Simulate the agent's reasoning flow on the seed graph."""

    def test_step1_lookup_huangrenxun(self, seeded_system):
        """Step 1: aad_lookup('黄仁勋') returns node with associations."""
        store, _ = seeded_system
        result = aad_lookup(store, "黄仁勋")
        assert result["ok"] is True
        node = result["node"]
        assert node["name"] == "黄仁勋"
        assert len(node["associations"]) == 2

    def test_step2_expand_from_huangrenxun_association(self, seeded_system):
        """Step 2: aad_expand on 黄仁勋's NVIDIA association finds NVIDIA."""
        store, index = seeded_system
        lookup_result = aad_lookup(store, "黄仁勋")
        nvidia_assoc = next(
            a for a in lookup_result["node"]["associations"]
            if "NVIDIA" in a["reason"]
        )

        expand_result = aad_expand(
            store, index, vector=nvidia_assoc["vector"], top_k=3
        )
        assert expand_result["ok"] is True
        result_names = {r["name"] for r in expand_result["results"]}
        assert "NVIDIA" in result_names

    def test_step3_get_nvidia_content(self, seeded_system):
        """Step 3: aad_get_content('NVIDIA') returns full text."""
        store, _ = seeded_system
        result = aad_get_content(store, "NVIDIA")
        assert result["ok"] is True
        assert "黄仁勋" in result["content"]
        assert "1993" in result["content"]

    def test_full_reasoning_chain(self, seeded_system):
        """Complete agent reasoning flow:
        1. Lookup 黄仁勋
        2. Expand NVIDIA association
        3. Get NVIDIA content
        4. Expand GPU association from NVIDIA
        5. Get GPU content
        """
        store, index = seeded_system

        # Step 1
        r1 = aad_lookup(store, "黄仁勋")
        assert r1["ok"]
        nvidia_assoc = next(
            a for a in r1["node"]["associations"]
            if "NVIDIA" in a["reason"]
        )

        # Step 2
        r2 = aad_expand(store, index, nvidia_assoc["vector"], top_k=3)
        assert r2["ok"]
        assert any(r["name"] == "NVIDIA" for r in r2["results"])

        # Step 3
        r3 = aad_get_content(store, "NVIDIA")
        assert r3["ok"]
        assert "GPU" in r3["content"]

        # Step 4
        r4 = aad_lookup(store, "NVIDIA")
        gpu_assoc = next(
            a for a in r4["node"]["associations"]
            if "GPU" in a["reason"]
        )

        # Step 5
        r5 = aad_expand(store, index, gpu_assoc["vector"], top_k=3)
        assert r5["ok"]
        assert any(r["name"] == "GPU" for r in r5["results"])

        # Step 6
        r6 = aad_get_content(store, "GPU")
        assert r6["ok"]
        assert "图形处理单元" in r6["content"]

    def test_reason_filter_finds_relevant_associations(self, seeded_system):
        """aad_expand with reason_filter filters results correctly."""
        store, index = seeded_system
        jensen = aad_lookup(store, "黄仁勋")
        nvidia_vec = next(
            a["vector"] for a in jensen["node"]["associations"]
            if "NVIDIA" in a["reason"]
        )
        result = aad_expand(
            store, index, nvidia_vec, top_k=5, reason_filter="创立"
        )
        assert result["ok"]
        nvidia_results = [r for r in result["results"] if r["name"] == "NVIDIA"]
        assert len(nvidia_results) > 0

    def test_error_propagation_not_found(self, seeded_system):
        """aad_lookup for nonexistent node returns ok=False."""
        store, _ = seeded_system
        result = aad_lookup(store, "QuantumComputing")
        assert result["ok"] is False
        assert "QuantumComputing" in result["error"]
        assert "Available nodes" in result["error"]

    def test_all_seed_nodes_are_reachable(self, seeded_system):
        """Every seed node can be found and its content retrieved."""
        store, _ = seeded_system
        for name in ["黄仁勋", "GPU", "NVIDIA"]:
            lookup = aad_lookup(store, name)
            assert lookup["ok"], f"aad_lookup({name!r}) failed"
            content = aad_get_content(store, name)
            assert content["ok"], f"aad_get_content({name!r}) failed"
            assert len(content["content"]) > 0
