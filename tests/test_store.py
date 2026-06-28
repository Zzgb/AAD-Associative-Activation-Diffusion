import pytest
import tempfile
from pathlib import Path

from aad.store import AADStore
from aad.models import Node, Association
from aad.errors import StorageError


@pytest.fixture
def tmp_store_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_store.jsonl"


@pytest.fixture
def store(tmp_store_path):
    return AADStore(str(tmp_store_path))


class TestAADStore:
    def test_empty_store_has_zero_length(self, store):
        assert len(store) == 0

    def test_put_and_get_node(self, store):
        node = Node(name="GPU", content="Graphics Processing Unit")
        store.put(node)
        assert len(store) == 1
        retrieved = store.get("GPU")
        assert retrieved is not None
        assert retrieved.name == "GPU"
        assert retrieved.content == "Graphics Processing Unit"

    def test_get_nonexistent_returns_none(self, store):
        assert store.get("missing") is None

    def test_get_content_returns_string(self, store):
        store.put(Node(name="GPU", content="hello"))
        assert store.get_content("GPU") == "hello"

    def test_get_content_nonexistent_returns_none(self, store):
        assert store.get_content("missing") is None

    def test_list_names(self, store):
        store.put(Node(name="A"))
        store.put(Node(name="B"))
        assert sorted(store.list_names()) == ["A", "B"]

    def test_contains(self, store):
        store.put(Node(name="X"))
        assert "X" in store
        assert "Y" not in store

    def test_iter_yields_all_nodes(self, store):
        store.put(Node(name="A", content="ca"))
        store.put(Node(name="B", content="cb"))
        names = {n.name for n in store}
        assert names == {"A", "B"}

    def test_put_overwrites_existing(self, store):
        store.put(Node(name="A", content="v1"))
        store.put(Node(name="A", content="v2"))
        assert store.get("A").content == "v2"
        assert len(store) == 1

    def test_delete_existing_returns_true(self, store):
        store.put(Node(name="X"))
        assert store.delete("X") is True
        assert "X" not in store
        assert len(store) == 0

    def test_delete_nonexistent_returns_false(self, store):
        assert store.delete("X") is False

    def test_clear_removes_all(self, store):
        store.put(Node(name="A"))
        store.put(Node(name="B"))
        store.clear()
        assert len(store) == 0

    def test_persistence_survives_reload(self, tmp_store_path):
        store1 = AADStore(str(tmp_store_path))
        store1.put(Node(name="P", content="persistent"))
        del store1

        store2 = AADStore(str(tmp_store_path))
        assert len(store2) == 1
        assert store2.get("P").content == "persistent"

    def test_persistence_handles_associations(self, tmp_store_path):
        store1 = AADStore(str(tmp_store_path))
        node = Node(
            name="GPU",
            content="Graphics",
            vector=[0.1, 0.2],
            associations=[Association(vector=[0.3, 0.4], reason="made by")],
        )
        store1.put(node)
        del store1

        store2 = AADStore(str(tmp_store_path))
        restored = store2.get("GPU")
        assert restored.vector == [0.1, 0.2]
        assert len(restored.associations) == 1
        assert restored.associations[0].reason == "made by"

    def test_put_empty_name_raises(self, store):
        with pytest.raises(StorageError, match="empty"):
            store.put(Node(name=""))
