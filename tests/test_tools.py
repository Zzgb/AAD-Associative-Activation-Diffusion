import pytest
import tempfile
import hashlib
from pathlib import Path
from unittest.mock import MagicMock

from aad.store import AADStore
from aad.vector_index import VectorIndex
from aad.session import SessionMemory
from aad.models import Node, Association
from aad.tools import (
    aad_lookup, aad_expand, aad_get_content,
    execute_tool, TOOL_SCHEMAS, _clear_refs,
)

DIM = 4


def _dummy_session():
    emb = MagicMock()
    emb.embed.return_value = [0.1] * DIM
    emb.embed_batch.return_value = [[0.1] * DIM]
    return SessionMemory(emb, dim=DIM)


@pytest.fixture(autouse=True)
def clear_refs():
    _clear_refs()
    yield


@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as tmpdir:
        s = AADStore(str(Path(tmpdir) / "test.jsonl"))
        yield s


@pytest.fixture
def index():
    return VectorIndex(dim=DIM)


@pytest.fixture
def session():
    return _dummy_session()


@pytest.fixture
def populated(store, index, session):
    node_a = Node(
        name="GPU", content="Graphics Processing Unit",
        vector=[1.0, 0.0, 0.0, 0.0],
        associations=[Association(vector=[0.0, 1.0, 0.0, 0.0], reason="manufactured by")],
    )
    node_b = Node(
        name="NVIDIA", content="NVIDIA Corporation",
        vector=[0.0, 1.0, 0.0, 0.0],
        associations=[Association(vector=[1.0, 0.0, 0.0, 0.0], reason="manufactures GPU")],
    )
    store.put(node_a)
    store.put(node_b)
    index.add("GPU", node_a.vector)
    index.add("NVIDIA", node_b.vector)
    _clear_refs()
    return store, index, session


class TestToolSchemas:
    def test_all_three_schemas_present(self):
        names = {s["function"]["name"] for s in TOOL_SCHEMAS}
        assert names == {"aad_lookup", "aad_expand", "aad_get_content"}

    def test_schemas_have_type_function(self):
        for schema in TOOL_SCHEMAS:
            assert schema["type"] == "function"
            assert "function" in schema

    def test_lookup_schema_requires_name(self):
        lookup = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "aad_lookup")
        assert "name" in lookup["function"]["parameters"]["required"]

    def test_expand_schema_requires_ref(self):
        expand = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "aad_expand")
        assert "ref" in expand["function"]["parameters"]["required"]


class TestAADLookup:
    def test_returns_node_from_longterm(self, populated):
        store, _, session = populated
        result = aad_lookup(store, session, "GPU")
        assert result["ok"] is True
        assert result["node"]["name"] == "GPU"
        assert result["source"] == "long_term"
        # Should have been mirrored
        assert session.is_visited("GPU")

    def test_returns_cached_from_session(self, populated):
        store, _, session = populated
        aad_lookup(store, session, "GPU")  # first: from longterm, mirrors
        result = aad_lookup(store, session, "GPU")  # second: from session
        assert result["ok"] is True
        assert result["source"] == "short_term"

    def test_returns_error_when_not_found(self, store, session):
        result = aad_lookup(store, session, "MISSING")
        assert result["ok"] is False


class TestAADExpand:
    def test_expand_from_ref(self, populated):
        store, index, session = populated
        lookup = aad_lookup(store, session, "GPU")
        ref = lookup["node"]["associations"][0]["ref"]
        result = aad_expand(store, index, session, ref=ref, top_k=2)
        assert result["ok"] is True
        names = {r["name"] for r in result["results"]}
        assert "NVIDIA" in names

    def test_expand_invalid_ref(self, populated):
        store, index, session = populated
        result = aad_expand(store, index, session, ref="bad", top_k=3)
        assert result["ok"] is False

    def test_expand_clamps_top_k(self, populated):
        store, index, session = populated
        lookup = aad_lookup(store, session, "GPU")
        ref = lookup["node"]["associations"][0]["ref"]
        result = aad_expand(store, index, session, ref=ref, top_k=100)
        assert result["ok"] is True


class TestAADGetContent:
    def test_returns_content(self, populated):
        store, _, session = populated
        result = aad_get_content(store, session, "GPU")
        assert result["ok"] is True
        assert "Graphics" in result["content"]

    def test_returns_error(self, store, session):
        result = aad_get_content(store, session, "MISSING")
        assert result["ok"] is False


class TestExecuteTool:
    def test_dispatches_lookup(self, populated):
        store, index, session = populated
        result = execute_tool(store, index, session, "aad_lookup", {"name": "GPU"})
        assert result["ok"] is True

    def test_dispatches_expand(self, populated):
        store, index, session = populated
        lookup = aad_lookup(store, session, "GPU")
        ref = lookup["node"]["associations"][0]["ref"]
        result = execute_tool(store, index, session, "aad_expand", {"ref": ref})
        assert result["ok"] is True

    def test_dispatches_get_content(self, populated):
        store, index, session = populated
        result = execute_tool(store, index, session, "aad_get_content", {"name": "GPU"})
        assert result["ok"] is True

    def test_unknown_tool(self, populated):
        store, index, session = populated
        result = execute_tool(store, index, session, "unknown", {})
        assert result["ok"] is False
