"""Short-term session memory as a full AAD knowledge graph.

SessionMemory mirrors long-term nodes on access and tracks message
history with linear-chain associations. Same Node structure as
long-term memory, lives in RAM, discarded on /quit.
"""

from aad.models import Node, Association
from aad.embedder import Embedder
from aad.vector_index import VectorIndex


class SessionMemory:
    """Session-scoped working memory graph.

    - Message nodes (msg_N): one per user input, linked linearly.
    - Mirror nodes: shallow copies of long-term nodes, content rewritable.
    - Own FAISS index for short-term vector search.
    """

    def __init__(self, embedder: Embedder, dim: int) -> None:
        self._nodes: dict[str, Node] = {}
        self._index = VectorIndex(dim)
        self._embedder = embedder
        self._msg_counter = 0
        self._last_msg_name: str | None = None
        # Track which long-term node names have been mirrored
        self._mirrored: set[str] = set()
        # Collect node names used in the current round (for inference linking)
        self._round_nodes: list[str] = []

    # ── message nodes ─────────────────────────────────────────────

    def add_message(self, user_input: str) -> str:
        """Create a msg_N node for user input. Returns the message name."""
        self._msg_counter += 1
        name = f"msg_{self._msg_counter}"
        vector = self._embedder.embed(user_input)

        node = Node(
            name=name,
            content=user_input,
            vector=vector,
        )

        # Link to previous message (linear chain)
        if self._last_msg_name:
            prev_vec = self._nodes[self._last_msg_name].vector
            node.associations.append(
                Association(vector=prev_vec, reason="对话顺序")
            )

        self._nodes[name] = node
        self._index.add(name, vector)
        self._last_msg_name = name
        return name

    # ── mirror long-term nodes ─────────────────────────────────────

    def mirror_longterm(self, lt_node: Node) -> Node:
        """Create a short-term mirror of a long-term node.

        Returns the mirror node. If already mirrored, returns existing.
        Content and associations can be rewritten later to fit current context.
        """
        if lt_node.name in self._mirrored:
            return self._nodes[lt_node.name]

        mirror = Node(
            name=lt_node.name,
            content=lt_node.content,
            vector=lt_node.vector,
            associations=list(lt_node.associations),  # shallow copy, rewritable
        )
        self._nodes[mirror.name] = mirror
        self._index.add(mirror.name, mirror.vector)
        self._mirrored.add(mirror.name)
        return mirror

    # ── linking ────────────────────────────────────────────────────

    def link_message_to_node(self, msg_name: str, node_name: str, reason: str) -> None:
        """Add association from a message node to a mirrored node."""
        msg = self._nodes.get(msg_name)
        target = self._nodes.get(node_name)
        if msg is None or target is None:
            return
        # Avoid duplicate links
        for a in msg.associations:
            if a.vector == target.vector and a.reason == reason:
                return
        msg.associations.append(Association(vector=target.vector, reason=reason))

    def begin_round(self) -> None:
        """Clear the round tracking list."""
        self._round_nodes.clear()

    def track_node(self, node_name: str) -> None:
        """Mark a node as used in the current reasoning round."""
        if node_name not in self._round_nodes:
            self._round_nodes.append(node_name)

    def commit_round(self, msg_name: str) -> None:
        """Link the current message to all nodes used this round."""
        for node_name in self._round_nodes:
            self.link_message_to_node(msg_name, node_name, "推理引用")
        self._round_nodes.clear()

    # ── lookup ─────────────────────────────────────────────────────

    def is_visited(self, name: str) -> bool:
        return name in self._nodes

    def get_node(self, name: str) -> Node | None:
        return self._nodes.get(name)

    def get_index(self) -> VectorIndex:
        return self._index

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def mirrored_count(self) -> int:
        return len(self._mirrored)

    # ── summary for LLM context ────────────────────────────────────

    def summary(self) -> str:
        """Compact session state for injection into system prompt."""
        lines = ["[短期记忆状态]"]
        lines.append(f"消息节点: {self._msg_counter} 条")
        if self._mirrored:
            lines.append(f"已加载长期节点: {', '.join(sorted(self._mirrored))}")
        if self._nodes:
            lines.append(f"短期节点总数: {self.node_count}")
        return "\n".join(lines)

    # ── lifecycle ──────────────────────────────────────────────────

    def clear(self) -> None:
        """Destroy all short-term memory. Called on /quit."""
        self._nodes.clear()
        self._index = VectorIndex(self._index._dim)
        self._mirrored.clear()
        self._msg_counter = 0
        self._last_msg_name = None
        self._round_nodes.clear()
