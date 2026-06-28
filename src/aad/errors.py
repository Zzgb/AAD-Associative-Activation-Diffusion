"""AAD exception hierarchy. All exceptions inherit from AADError."""


class AADError(Exception):
    """Base exception for all AAD errors."""
    pass


class NodeNotFoundError(AADError):
    """Raised when a node is not found by name."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Node not found: {name!r}")


class NodeAlreadyExistsError(AADError):
    """Raised when attempting to create a node that already exists."""

    def __init__(self, name: str) -> None:
        self.name = name
        super().__init__(f"Node already exists: {name!r}")


class EmbeddingError(AADError):
    """Raised when embedding generation fails."""

    def __init__(self, message: str) -> None:
        super().__init__(f"Embedding error: {message}")


class StorageError(AADError):
    """Raised when persistence operations fail."""

    def __init__(self, message: str) -> None:
        super().__init__(f"Storage error: {message}")


class VectorIndexError(AADError):
    """Raised when vector index operations fail."""

    def __init__(self, message: str) -> None:
        super().__init__(f"Vector index error: {message}")
