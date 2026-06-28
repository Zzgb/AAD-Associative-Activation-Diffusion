import pytest
from aad.models import Node, Association


class TestAssociation:
    def test_creation(self):
        a = Association(vector=[1.0, 2.0, 3.0], reason="causes")
        assert a.vector == [1.0, 2.0, 3.0]
        assert a.reason == "causes"

    def test_defaults(self):
        a = Association()
        assert a.vector == []
        assert a.reason == ""

    def test_serialization_roundtrip(self):
        a = Association(vector=[1.0, 2.0], reason="test")
        data = a.model_dump()
        restored = Association.model_validate(data)
        assert restored == a


class TestNode:
    def test_creation_minimal(self):
        n = Node(name="test")
        assert n.name == "test"
        assert n.content == ""
        assert n.vector == []
        assert n.associations == []

    def test_creation_full(self):
        n = Node(
            name="GPU",
            content="Graphics Processing Unit",
            vector=[0.1, 0.2, 0.3],
            associations=[
                Association(vector=[0.4, 0.5, 0.6], reason="manufactured by")
            ],
        )
        assert n.name == "GPU"
        assert len(n.associations) == 1
        assert n.associations[0].reason == "manufactured by"

    def test_serialization_roundtrip(self):
        n = Node(
            name="GPU",
            content="Graphics Processing Unit",
            vector=[0.1, 0.2],
            associations=[Association(vector=[0.3, 0.4], reason="related")],
        )
        data = n.model_dump()
        restored = Node.model_validate(data)
        assert restored == n

    def test_name_is_required(self):
        with pytest.raises(ValueError):
            Node()

    def test_vector_defaults_to_empty_list(self):
        n = Node(name="x")
        assert n.vector == []
        assert isinstance(n.vector, list)
