"""AAD tool definitions (OpenAI function-calling format) and implementations.

Each tool returns a dict with an "ok" key:
  - ok=True  → result contains the requested data
  - ok=False → error contains a human-readable message

Key design: raw vectors are NEVER passed to the LLM. A session-level
ref table maps short string refs to full vectors. The LLM only sees
ref strings; resolve happens server-side in execute_tool.
"""

from typing import Any

from aad.store import AADStore
from aad.vector_index import VectorIndex

# ──────────────────────────────────────────────────────────────────
# Vector Ref Table (session-scoped, prevents LLM vector truncation)
# ──────────────────────────────────────────────────────────────────

# ref → full vector
_ref_table: dict[str, list[float]] = {}
_ref_counter: int = 0


def _make_ref(vector: list[float]) -> str:
    """Store a vector and return a short ref string for the LLM."""
    global _ref_counter
    ref = f"v{_ref_counter}"
    _ref_counter += 1
    _ref_table[ref] = vector
    return ref


def _resolve_ref(ref: str) -> list[float] | None:
    """Look up a ref in the session table."""
    return _ref_table.get(ref)


def _clear_refs() -> None:
    """Reset the ref table (call between sessions)."""
    global _ref_counter
    _ref_table.clear()
    _ref_counter = 0


def _assoc_to_llm(assoc) -> dict[str, Any]:
    """Convert an Association to LLM-safe dict (ref instead of vector)."""
    return {
        "ref": _make_ref(assoc.vector),
        "reason": assoc.reason,
    }


# ──────────────────────────────────────────────────────────────────
# Tool Schemas (OpenAI / DeepSeek function-calling format)
# ──────────────────────────────────────────────────────────────────

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "aad_lookup",
            "description": (
                "在 AAD 知识图谱中通过名称精确查找节点。"
                "返回节点内容以及关联列表（每个关联有一个 ref 引用和原因描述）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "节点名称（如 'GPU', 'NVIDIA', '黄仁勋'）",
                    }
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
                "从一个关联 ref 出发，寻找知识图谱中的关联节点。"
                "传入 aad_lookup 返回的 associations 中某个关联的 ref。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ref": {
                        "type": "string",
                        "description": "关联引用 ID（从 aad_lookup 返回的 associations 中的 ref 字段获取）",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "最大返回结果数（默认 3）",
                        "default": 3,
                    },
                },
                "required": ["ref"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "aad_get_content",
            "description": (
                "获取节点的完整文字内容。当你通过 aad_lookup 或 aad_expand "
                "发现了节点名称，需要其详细描述时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "节点名称",
                    }
                },
                "required": ["name"],
            },
        },
    },
]

# ──────────────────────────────────────────────────────────────────
# Tool Implementations
# ──────────────────────────────────────────────────────────────────


def aad_lookup(store: AADStore, name: str) -> dict[str, Any]:
    """Look up a node by exact name. Returns node with ref-based associations."""
    node = store.get(name)
    if node is None:
        return {
            "ok": False,
            "error": f"节点未找到: {name!r}。已知节点: {store.list_names()}",
        }
    return {
        "ok": True,
        "node": {
            "name": node.name,
            "content": node.content,
            "associations": [_assoc_to_llm(a) for a in node.associations],
        },
    }


def aad_expand(
    store: AADStore,
    index: VectorIndex,
    ref: str,
    top_k: int = 3,
) -> dict[str, Any]:
    """Search for nodes near the vector referenced by ref."""
    vector = _resolve_ref(ref)
    if vector is None:
        return {"ok": False, "error": f"无效的引用: {ref!r}，请使用 aad_lookup 返回的 ref"}

    top_k = max(1, min(top_k, 20))
    raw_results = index.search(vector, top_k=top_k)
    if not raw_results:
        return {"ok": True, "results": []}

    results: list[dict[str, Any]] = []
    for name, score in raw_results:
        node = store.get(name)
        if node is None:
            continue
        results.append({
            "name": node.name,
            "content": node.content,
            "score": round(float(score), 4),
            "associations": [_assoc_to_llm(a) for a in node.associations],
        })

    return {"ok": True, "results": results}


def aad_get_content(store: AADStore, name: str) -> dict[str, Any]:
    """Retrieve only the content text of a node."""
    content = store.get_content(name)
    if content is None:
        return {
            "ok": False,
            "error": f"节点未找到: {name!r}。已知节点: {store.list_names()}",
        }
    return {"ok": True, "name": name, "content": content}


# ──────────────────────────────────────────────────────────────────
# Tool Dispatcher
# ──────────────────────────────────────────────────────────────────


def execute_tool(
    store: AADStore,
    index: VectorIndex,
    tool_name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:
    """Dispatch a tool call by name. Resolves refs server-side."""
    if tool_name == "aad_lookup":
        return aad_lookup(store, name=arguments["name"])
    elif tool_name == "aad_expand":
        return aad_expand(
            store=store,
            index=index,
            ref=arguments["ref"],
            top_k=arguments.get("top_k", 3),
        )
    elif tool_name == "aad_get_content":
        return aad_get_content(store, name=arguments["name"])
    else:
        return {"ok": False, "error": f"未知工具: {tool_name!r}"}
