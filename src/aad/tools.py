"""AAD tool definitions (OpenAI function-calling format) and implementations.

Each tool returns a dict with an "ok" key:
  - ok=True  → result contains the requested data
  - ok=False → error contains a human-readable message

Tool schemas are exposed as TOOL_SCHEMAS for consumption by LLM SDKs.
"""

from typing import Any

from aad.store import AADStore
from aad.vector_index import VectorIndex

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
                "返回完整节点，包含内容、嵌入向量和所有关联。"
                "当你已知某个概念名称需要查找其详细信息时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "要查找的节点名称（如 'GPU', 'NVIDIA', '黄仁勋'）",
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
                "从给定的嵌入向量出发，进行近似最近邻搜索，寻找知识图谱中的关联节点。"
                "可选的 reason_filter 仅保留关联原因中包含指定文本的结果。"
                "用于从概念向外发散探索图谱。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "vector": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "查询嵌入向量（通常取自某个节点的关联向量或自身向量）",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "最大返回结果数（默认 3）",
                        "default": 3,
                    },
                    "reason_filter": {
                        "type": "string",
                        "description": "可选。仅返回关联原因中包含此文本的节点",
                    },
                },
                "required": ["vector"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "aad_get_content",
            "description": (
                "仅获取节点的纯文本内容，不返回向量和关联。"
                "当你已知节点存在，只需其文字描述来构建回答上下文时使用。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "要获取内容的节点名称",
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
    """Look up a node by exact name. Returns full node data."""
    node = store.get(name)
    if node is None:
        return {
            "ok": False,
            "error": f"Node not found: {name!r}. Available nodes: {store.list_names()}",
        }
    return {
        "ok": True,
        "node": node.model_dump(),
    }


def aad_expand(
    store: AADStore,
    index: VectorIndex,
    vector: list[float],
    top_k: int = 3,
    reason_filter: str | None = None,
) -> dict[str, Any]:
    """Search for nodes near the given vector, optionally filtered by reason."""
    top_k = max(1, min(top_k, 20))

    raw_results = index.search(vector, top_k=top_k)
    if not raw_results:
        return {"ok": True, "results": []}

    results: list[dict[str, Any]] = []
    for name, score in raw_results:
        node = store.get(name)
        if node is None:
            continue

        matching_associations: list[dict[str, Any]] = []
        if reason_filter is not None:
            for assoc in node.associations:
                if reason_filter.lower() in assoc.reason.lower():
                    matching_associations.append(assoc.model_dump())

        entry: dict[str, Any] = {
            "name": node.name,
            "content": node.content,
            "score": round(float(score), 4),
        }
        if reason_filter is not None:
            entry["matching_associations"] = matching_associations
        else:
            entry["associations"] = [a.model_dump() for a in node.associations]

        results.append(entry)

    return {"ok": True, "results": results}


def aad_get_content(store: AADStore, name: str) -> dict[str, Any]:
    """Retrieve only the content text of a node."""
    content = store.get_content(name)
    if content is None:
        return {
            "ok": False,
            "error": f"Node not found: {name!r}. Available nodes: {store.list_names()}",
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
    """Dispatch a tool call by name. Returns the tool's result dict."""
    if tool_name == "aad_lookup":
        return aad_lookup(store, name=arguments["name"])
    elif tool_name == "aad_expand":
        return aad_expand(
            store=store,
            index=index,
            vector=arguments["vector"],
            top_k=arguments.get("top_k", 3),
            reason_filter=arguments.get("reason_filter"),
        )
    elif tool_name == "aad_get_content":
        return aad_get_content(store, name=arguments["name"])
    else:
        return {"ok": False, "error": f"Unknown tool: {tool_name!r}"}
