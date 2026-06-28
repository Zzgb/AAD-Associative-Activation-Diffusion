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
)

DIM = 4


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
    """Populate store and index with test nodes."""
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
        lookup = next(
            s for s in TOOL_SCHEMAS if s["function"]["name"] == "aad_lookup"
        )
        assert "name" in lookup["function"]["parameters"]["required"]

    def test_expand_schema_requires_vector(self):
        expand = next(
            s for s in TOOL_SCHEMAS if s["function"]["name"] == "aad_expand"
        )
        assert "vector" in expand["function"]["parameters"]["required"]


class TestAADLookup:
    def test_returns_node_when_found(self, populated):
        store, _ = populated
        result = aad_lookup(store, "GPU")
        assert result["ok"] is True
        assert result["node"]["name"] == "GPU"
        assert result["node"]["content"] == "Graphics Processing Unit"
        assert len(result["node"]["associations"]) == 1

    def test_returns_error_when_not_found(self, store):
        result = aad_lookup(store, "MISSING")
        assert result["ok"] is False
        assert "MISSING" in result["error"]


class TestAADExpand:
    def test_expand_finds_related_nodes(self, populated):
        store, index = populated
        result = aad_expand(store, index, vector=[1.0, 0.0, 0.0, 0.0], top_k=2)
        assert result["ok"] is True
        assert len(result["results"]) == 2
        assert result["results"][0]["name"] == "GPU"

    def test_expand_with_reason_filter(self, populated):
        store, index = populated
        result = aad_expand(
            store, index, vector=[1.0, 0.0, 0.0, 0.0],
            top_k=2, reason_filter="manufactured",
        )
        assert result["ok"] is True
        gpu_result = next(r for r in result["results"] if r["name"] == "GPU")
        assert len(gpu_result["matching_associations"]) == 1
        assert "manufactured by" in gpu_result["matching_associations"][0]["reason"]

    def test_expand_clamps_top_k(self, populated):
        store, index = populated
        result = aad_expand(store, index, vector=[1.0, 0.0, 0.0, 0.0], top_k=100)
        assert result["ok"] is True

    def test_expand_empty_index_returns_empty(self, store, index):
        result = aad_expand(store, index, vector=[1.0, 0.0, 0.0, 0.0])
        assert result["ok"] is True
        assert result["results"] == []


class TestAADGetContent:
    def test_returns_content_when_found(self, populated):
        store, _ = populated
        result = aad_get_content(store, "GPU")
        assert result["ok"] is True
        assert result["content"] == "Graphics Processing Unit"

    def test_returns_error_when_not_found(self, store):
        result = aad_get_content(store, "MISSING")
        assert result["ok"] is False
        assert "MISSING" in result["error"]


class TestExecuteTool:
    def test_dispatches_aad_lookup(self, populated):
        store, index = populated
        result = execute_tool(store, index, "aad_lookup", {"name": "GPU"})
        assert result["ok"] is True

    def test_dispatches_aad_expand(self, populated):
        store, index = populated
        result = execute_tool(
            store, index, "aad_expand", {"vector": [1.0, 0.0, 0.0, 0.0]}
        )
        assert result["ok"] is True

    def test_dispatches_aad_get_content(self, populated):
        store, index = populated
        result = execute_tool(store, index, "aad_get_content", {"name": "GPU"})
        assert result["ok"] is True

    def test_unknown_tool_returns_error(self, populated):
        store, index = populated
        result = execute_tool(store, index, "unknown_tool", {})
        assert result["ok"] is False
        assert "unknown_tool" in result["error"]
