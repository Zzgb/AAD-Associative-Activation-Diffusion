import pytest
from aad.vector_index import VectorIndex
from aad.errors import VectorIndexError

DIM = 4


@pytest.fixture
def index():
    return VectorIndex(dim=DIM)


class TestVectorIndex:
    def test_init_requires_positive_dim(self):
        with pytest.raises(VectorIndexError):
            VectorIndex(dim=0)
        with pytest.raises(VectorIndexError):
            VectorIndex(dim=-1)

    def test_empty_index_search_returns_empty(self, index):
        result = index.search([1.0, 0.0, 0.0, 0.0])
        assert result == []

    def test_add_and_search_single(self, index):
        index.add("A", [1.0, 0.0, 0.0, 0.0])
        results = index.search([1.0, 0.0, 0.0, 0.0], top_k=3)
        assert len(results) == 1
        assert results[0][0] == "A"

    def test_search_returns_closest(self, index):
        index.add("near", [1.0, 0.0, 0.0, 0.0])
        index.add("far", [0.0, 0.0, 0.0, 1.0])
        results = index.search([1.0, 0.0, 0.0, 0.0], top_k=2)
        assert results[0][0] == "near"
        assert results[1][0] == "far"
        assert results[0][1] > results[1][1]

    def test_dimension_mismatch_raises(self, index):
        with pytest.raises(VectorIndexError, match="dimension mismatch"):
            index.add("A", [1.0, 2.0])
        # Need at least one node for search to reach validation
        index.add("B", [1.0, 0.0, 0.0, 0.0])
        with pytest.raises(VectorIndexError, match="dimension mismatch"):
            index.search([1.0, 2.0])

    def test_remove(self, index):
        index.add("A", [1.0, 0.0, 0.0, 0.0])
        assert len(index) == 1
        index.remove("A")
        assert len(index) == 0
        assert index.search([1.0, 0.0, 0.0, 0.0]) == []

    def test_remove_nonexistent_no_error(self, index):
        index.remove("missing")

    def test_update_replaces_vector(self, index):
        index.add("A", [1.0, 0.0, 0.0, 0.0])
        index.add("A", [0.0, 0.0, 0.0, 1.0])
        results = index.search([0.0, 0.0, 0.0, 1.0], top_k=1)
        assert results[0][0] == "A"

    def test_rebuild_from_nodes(self, index):
        from aad.models import Node

        nodes = {
            "A": Node(name="A", content="a", vector=[1.0, 0.0, 0.0, 0.0]),
            "B": Node(name="B", content="b", vector=[0.0, 1.0, 0.0, 0.0]),
        }
        index.rebuild(nodes)
        assert len(index) == 2
        results = index.search([1.0, 0.0, 0.0, 0.0])
        assert results[0][0] == "A"
