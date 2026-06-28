from aad.errors import (
    AADError,
    NodeNotFoundError,
    NodeAlreadyExistsError,
    EmbeddingError,
    StorageError,
    VectorIndexError,
)


def test_aad_error_is_base():
    assert issubclass(NodeNotFoundError, AADError)
    assert issubclass(EmbeddingError, AADError)


def test_node_not_found_error_str():
    err = NodeNotFoundError("test_node")
    assert "test_node" in str(err)
    assert err.name == "test_node"


def test_node_already_exists_error_str():
    err = NodeAlreadyExistsError("dup")
    assert "dup" in str(err)
    assert err.name == "dup"


def test_embedding_error_str():
    err = EmbeddingError("API timeout")
    assert "API timeout" in str(err)


def test_storage_error_str():
    err = StorageError("disk full")
    assert "disk full" in str(err)


def test_vector_index_error_str():
    err = VectorIndexError("dimension mismatch")
    assert "dimension mismatch" in str(err)
