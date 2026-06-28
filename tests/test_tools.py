import pytest
import tempfile
from pathlib import Path

from aad.store import AADStore
from aad.vector_index import VectorIndex
from aad.models import Node, Association
from aad.tools import (
    aad_lookup,
    aad_expand,
    aad_get_content,
    execute_tool,
    TOOL_SCHEMAS,
    _clear_refs,
)

DIM = 4


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
def populated(store, index):
    node_a = Node(
        name="GPU",
        content="Graphics Processing Unit",
        vector=[1.0, 0.0, 0.0, 0.0],
        associations=[
            Association(vector=[0.0, 1.0, 0.0, 0.0], reason="manufactured by")
        ],
    )
    node_b = Node(
        name="NVIDIA",
        content="NVIDIA Corporation",
        vector=[0.0, 1.0, 0.0, 0.0],
        associations=[
            Association(vector=[1.0, 0.0, 0.0, 0.0], reason="manufactures GPU")
        ],
    )
    store.put(node_a)
    store.put(node_b)
    index.add("GPU", node_a.vector)
    index.add("NVIDIA", node_b.vector)
    _clear_refs()
    return store, index


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
    def test_returns_node_when_found(self, populated):
        store, _ = populated
        result = aad_lookup(store, "GPU")
        assert result["ok"] is True
        assert result["node"]["name"] == "GPU"
        assert result["node"]["content"] == "Graphics Processing Unit"
        assert len(result["node"]["associations"]) == 1
        # associations use ref, not raw vector
        assert "ref" in result["node"]["associations"][0]
        assert "reason" in result["node"]["associations"][0]
        assert "vector" not in result["node"]["associations"][0]

    def test_returns_error_when_not_found(self, store):
        result = aad_lookup(store, "MISSING")
        assert result["ok"] is False


class TestAADExpand:
    def test_expand_from_ref(self, populated):
        store, index = populated
        # First lookup to get a ref
        lookup = aad_lookup(store, "GPU")
        ref = lookup["node"]["associations"][0]["ref"]
        # Then expand from that ref
        result = aad_expand(store, index, ref=ref, top_k=2)
        assert result["ok"] is True
        assert len(result["results"]) >= 1
        names = {r["name"] for r in result["results"]}
        assert "NVIDIA" in names

    def test_expand_invalid_ref(self, populated):
        store, index = populated
        result = aad_expand(store, index, ref="bad_ref", top_k=3)
        assert result["ok"] is False
        assert "bad_ref" in result["error"]

    def test_expand_clamps_top_k(self, populated):
        store, index = populated
        lookup = aad_lookup(store, "GPU")
        ref = lookup["node"]["associations"][0]["ref"]
        result = aad_expand(store, index, ref=ref, top_k=100)
        assert result["ok"] is True


class TestAADGetContent:
    def test_returns_content_when_found(self, populated):
        store, _ = populated
        result = aad_get_content(store, "GPU")
        assert result["ok"] is True
        assert result["content"] == "Graphics Processing Unit"

    def test_returns_error_when_not_found(self, store):
        result = aad_get_content(store, "MISSING")
        assert result["ok"] is False


class TestExecuteTool:
    def test_dispatches_aad_lookup(self, populated):
        store, index = populated
        result = execute_tool(store, index, "aad_lookup", {"name": "GPU"})
        assert result["ok"] is True

    def test_dispatches_aad_expand(self, populated):
        store, index = populated
        lookup = aad_lookup(store, "GPU")
        ref = lookup["node"]["associations"][0]["ref"]
        result = execute_tool(store, index, "aad_expand", {"ref": ref})
        assert result["ok"] is True

    def test_dispatches_aad_get_content(self, populated):
        store, index = populated
        result = execute_tool(store, index, "aad_get_content", {"name": "GPU"})
        assert result["ok"] is True

    def test_unknown_tool_returns_error(self, populated):
        store, index = populated
        result = execute_tool(store, index, "unknown_tool", {})
        assert result["ok"] is False
