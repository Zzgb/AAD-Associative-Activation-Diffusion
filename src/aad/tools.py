"""AAD tool definitions (OpenAI function-calling format) and implementations.

Each tool returns a dict with an "ok" key.
Vectors never passed to LLM — ref table maps short string refs to full vectors.
SessionMemory: auto-mirror, dual-index search, inference linking.
"""

from typing import Any

from aad.store import AADStore
from aad.vector_index import VectorIndex
from aad.session import SessionMemory

# ──────────────────────────────────────────────────────────────────
# Vector Ref Table
# ──────────────────────────────────────────────────────────────────

_ref_table: dict[str, list[float]] = {}
_ref_counter: int = 0


def _make_ref(vector: list[float]) -> str:
    global _ref_counter
    ref = f"v{_ref_counter}"
    _ref_counter += 1
    _ref_table[ref] = vector
    return ref


def _resolve_ref(ref: str) -> list[float] | None:
    return _ref_table.get(ref)


def _clear_refs() -> None:
    global _ref_counter
    _ref_table.clear()
    _ref_counter = 0


def _assoc_to_llm(assoc) -> dict[str, Any]:
    return {"ref": _make_ref(assoc.vector), "reason": assoc.reason}


# ──────────────────────────────────────────────────────────────────
# Tool Schemas
# ──────────────────────────────────────────────────────────────────

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "aad_lookup",
            "description": (
                "在 AAD 知识图谱中通过名称精确查找节点。先在短期记忆中查找，"
                "未命中再查长期记忆。返回节点内容以及关联列表。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "节点名称"},
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "aad_expand",
            "description": (
                "从一个关联 ref 出发，搜索短期+长期记忆中的关联节点。"
                "传入 aad_lookup 返回的 associations 中某个关联的 ref。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {"type": "string", "description": "关联引用 ID"},
                    "top_k": {"type": "integer", "description": "最大返回结果数（默认 3）", "default": 3},
                },
                "required": ["ref"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "aad_get_content",
            "description": "获取节点的完整文字内容。先查短期记忆，再查长期记忆。",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "节点名称"},
                },
                "required": ["name"],
            },
        },
    },
]

# ──────────────────────────────────────────────────────────────────
# Tool Implementations
# ──────────────────────────────────────────────────────────────────


def _node_to_llm(node) -> dict[str, Any]:
    """Serialize a Node for LLM consumption (ref-based associations)."""
    return {
        "name": node.name,
        "content": node.content,
        "associations": [_assoc_to_llm(a) for a in node.associations],
    }


def aad_lookup(
    store: AADStore,
    session: SessionMemory,
    name: str,
) -> dict[str, Any]:
    """Look up a node: short-term → long-term. Auto-mirror on long-term hit."""
    # 1. Check short-term memory
    sm_node = session.get_node(name)
    if sm_node is not None:
        session.track_node(name)
        return {"ok": True, "node": _node_to_llm(sm_node), "source": "short_term"}

    # 2. Check long-term memory
    lt_node = store.get(name)
    if lt_node is None:
        lt_names = store.list_names()
        sm_names = [n for n in [name] if session.is_visited(n)]  # for error hint
        all_names = sorted(set(lt_names) | {n for n in sm_names})
        return {"ok": False, "error": f"节点未找到: {name!r}。已知节点: {all_names}"}

    # 3. Auto-mirror
    mirror = session.mirror_longterm(lt_node)
    session.track_node(name)
    return {"ok": True, "node": _node_to_llm(mirror), "source": "long_term"}


def aad_expand(
    store: AADStore,
    index: VectorIndex,
    session: SessionMemory,
    ref: str,
    top_k: int = 3,
) -> dict[str, Any]:
    """Search short-term + long-term indexes. Auto-mirror long-term results."""
    vector = _resolve_ref(ref)
    if vector is None:
        return {"ok": False, "error": f"无效的引用: {ref!r}"}

    top_k = max(1, min(top_k, 20))
    sm_index = session.get_index()
    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    # 1. Search short-term index
    for name, score in sm_index.search(vector, top_k=top_k):
        if name in seen:
            continue
        seen.add(name)
        node = session.get_node(name)
        if node is None:
            node = store.get(name)
        if node is not None:
            results.append({
                "name": node.name,
                "content": node.content,
                "score": round(float(score), 4),
                "source": "short_term",
                "associations": [_assoc_to_llm(a) for a in node.associations],
            })

    # 2. Search long-term index
    for name, score in index.search(vector, top_k=top_k):
        if name in seen:
            continue
        seen.add(name)
        lt_node = store.get(name)
        if lt_node is None:
            continue
        # Auto-mirror
        session.mirror_longterm(lt_node)
        session.track_node(name)
        results.append({
            "name": lt_node.name,
            "content": lt_node.content,
            "score": round(float(score), 4),
            "source": "long_term",
            "associations": [_assoc_to_llm(a) for a in lt_node.associations],
        })

    return {"ok": True, "results": results}


def aad_get_content(
    store: AADStore,
    session: SessionMemory,
    name: str,
) -> dict[str, Any]:
    """Get full content: short-term first, then long-term."""
    # Short-term
    sm_node = session.get_node(name)
    if sm_node is not None:
        return {"ok": True, "name": name, "content": sm_node.content, "source": "short_term"}

    # Long-term
    content = store.get_content(name)
    if content is None:
        return {"ok": False, "error": f"节点未找到: {name!r}。已知节点: {store.list_names()}"}
    return {"ok": True, "name": name, "content": content, "source": "long_term"}


# ──────────────────────────────────────────────────────────────────
# Tool Dispatcher
# ──────────────────────────────────────────────────────────────────


def execute_tool(
    store: AADStore,
    index: VectorIndex,
    session: SessionMemory,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Dispatch a tool call. Tracks nodes used for inference linking."""
    if tool_name == "aad_lookup":
        return aad_lookup(store, session, name=arguments["name"])
    elif tool_name == "aad_expand":
        return aad_expand(
            store=store, index=index, session=session,
            ref=arguments["ref"], top_k=arguments.get("top_k", 3),
        )
    elif tool_name == "aad_get_content":
        return aad_get_content(store, session, name=arguments["name"])
    else:
        return {"ok": False, "error": f"未知工具: {tool_name!r}"}
