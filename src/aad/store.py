"""In-memory HashMap store with JSONL persistence for AAD nodes."""

import json
from pathlib import Path
from collections.abc import Iterator

from aad.models import Node
from aad.errors import StorageError


class AADStore:
    """In-memory store for AAD nodes with JSONL file persistence.

    Nodes are keyed by name. On initialization, loads existing nodes
    from a JSONL file. Every mutation persists the full dataset.
    """

    def __init__(self, filepath: str) -> None:
        self._filepath = Path(filepath)
        self._nodes: dict[str, Node] = {}
        self._load()

    # ── public read API ──────────────────────────────────────────

    def get(self, name: str) -> Node | None:
        """Return the Node with the given name, or None if not found."""
        return self._nodes.get(name)

    def get_content(self, name: str) -> str | None:
        """Return just the content field of a node, or None if not found."""
        node = self._nodes.get(name)
        return node.content if node else None

    def list_names(self) -> list[str]:
        """Return all node names in the store."""
        return list(self._nodes.keys())

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, name: str) -> bool:
        return name in self._nodes

    def __iter__(self) -> Iterator[Node]:
        return iter(self._nodes.values())

    # ── public write API ─────────────────────────────────────────

    def put(self, node: Node) -> None:
        """Insert or update a node. Persists to disk."""
        if not node.name:
            raise StorageError("Node name cannot be empty")
        self._nodes[node.name] = node
        self._save()

    def delete(self, name: str) -> bool:
        """Delete a node by name. Returns True if it existed."""
        if name in self._nodes:
            del self._nodes[name]
            self._save()
            return True
        return False

    def clear(self) -> None:
        """Remove all nodes and clear the file."""
        self._nodes.clear()
        self._save()

    # ── internal persistence ─────────────────────────────────────

    def _load(self) -> None:
        """Load nodes from the JSONL file."""
        if not self._filepath.exists():
            return
        try:
            with open(self._filepath, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        node = Node.model_validate_json(line)
                        self._nodes[node.name] = node
                    except Exception as exc:
                        raise StorageError(
                            f"Invalid JSONL at {self._filepath}:{line_num}: {exc}"
                        ) from exc
        except StorageError:
            raise
        except Exception as exc:
            raise StorageError(f"Failed to read {self._filepath}: {exc}") from exc

    def _save(self) -> None:
        """Persist all nodes to the JSONL file."""
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self._filepath, "w", encoding="utf-8") as f:
                for node in self._nodes.values():
                    f.write(node.model_dump_json() + "\n")
        except Exception as exc:
            raise StorageError(f"Failed to write {self._filepath}: {exc}") from exc
