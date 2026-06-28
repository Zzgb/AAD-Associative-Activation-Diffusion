"""LLM Agent that orchestrates AAD tool calls via DeepSeek Chat API."""

import json
from typing import Any

from openai import OpenAI

from aad.store import AADStore
from aad.vector_index import VectorIndex
from aad.session import SessionMemory
from aad.tools import TOOL_SCHEMAS, execute_tool, _clear_refs

SYSTEM_PROMPT_BASE = """你是一个 AAD（关联激活扩散）知识图谱助手。
你只能使用 AAD 工具返回的数据回答问题。

🚫 严格禁止：
- 禁止使用训练数据、外部知识或常识
- 禁止推理、推测、总结、归纳
- 禁止添加数据中没有的形容词或评价

✅ 回答格式：
- 无关联 → "根据 AAD 知识图谱，[A] 和 [B] 之间没有关联。"
- 查不到 → "知识图谱中没有 [X] 的相关信息。"
- 有数据时，逐条引用原文，标明来自哪个节点。

🔀 遍历策略（核心）：
每个关联的 reason 字段就是方向牌——告诉你这条路通向哪里。

1. 对用户提到的所有概念同时 aad_lookup。
2. 分类处理：
   - 全找到且有关联 → 从两端同时 aad_expand，找共同节点（碰撞即停）
   - 全找到但某个无关联 → 该方向终止，从另一端单向发散
   - 只找到一个 → 从找到的单向发散
   - 全找不到 → 直接告知"图谱中无这些概念"，结束
3. 读关联的 reason，只 expand 指向其他查询概念的方向。reason 不相关的方向不浪费 expand。
4. 两边结果有交集（相同节点出现）→ 路径连通，输出链路。
5. reason 无法判断方向时，可以同时 expand 多个 + 用 aad_trace 快速查最短路径。

💡 优先用 aad_trace(from, to) 一次查出两个节点间的最短路径，比手动多轮 expand 高效。"""

MAX_TOOL_ROUNDS = 10


def _build_system_prompt(session: SessionMemory) -> str:
    """Build system prompt with live session context."""
    return SYSTEM_PROMPT_BASE + "\n\n" + session.summary()


def _format_result(result: dict[str, Any]) -> str:
    """Format a tool result for verbose logging."""
    ok = result.get("ok", False)
    if not ok:
        return f"✗ {result.get('error', 'unknown error')}"

    src = result.get("source", "?")
    if "node" in result:
        node = result["node"]
        assocs = node.get("associations", [])
        assoc_strs = [f"  ref={a['ref']} → [{a['reason']}]" for a in assocs]
        return (
            f"✓ [{src}] node: {node['name']}\n"
            f"  content: {node.get('content','')[:100]}...\n"
            f"  associations ({len(assocs)}):\n" + "\n".join(assoc_strs)
        )

    if "results" in result:
        items = []
        for r in result["results"]:
            assocs = r.get("associations", [])
            refs = ", ".join(a.get("ref", "?") for a in assocs)
            items.append(
                f"  [{r.get('source','?')}] {r['name']} (score={r.get('score','?')}) "
                f"— {r.get('content','')[:60]}... [refs: {refs}]"
            )
        return f"✓ {len(items)} results:\n" + "\n".join(items)

    if "content" in result:
        return f"✓ [{src}] content of '{result['name']}': {result['content'][:120]}..."

    if "path" in result:
        depth = result.get("depth", "?")
        msg = result.get("message", "")
        return f"✓ trace depth={depth}: {msg}"

    return json.dumps(result, ensure_ascii=False, indent=2)[:300]


class AADAgent:
    """Orchestrates AAD tool calls via DeepSeek Chat API."""

    def __init__(
        self,
        store: AADStore,
        index: VectorIndex,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        verbose: bool = True,
    ) -> None:
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is required")
        self._store = store
        self._index = index
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model
        self._verbose = verbose

    def run(self, user_query: str, session: SessionMemory | None = None) -> str:
        _clear_refs()

        has_session = session is not None

        # Create message node
        if has_session:
            msg_name = session.add_message(user_query)
            session.begin_round()

        # Build messages with dynamic system prompt
        system_content = _build_system_prompt(session) if has_session else SYSTEM_PROMPT_BASE
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_query},
        ]

        for round_num in range(1, MAX_TOOL_ROUNDS + 1):
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
            )

            msg = response.choices[0].message

            if not msg.tool_calls:
                if self._verbose:
                    print(f"\n{'─'*50}")
                    print(f"[轮次 {round_num}] LLM 决定停止，输出最终答案")
                    print(f"{'─'*50}\n")
                # Commit inference links
                if has_session:
                    session.commit_round(msg_name)
                return msg.content or ""

            if self._verbose:
                print(f"\n{'─'*50}")
                print(f"[轮次 {round_num}] LLM 调用 {len(msg.tool_calls)} 个工具")

            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            })

            for tc in msg.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                if self._verbose:
                    print(f"  → {tc.function.name}({json.dumps(arguments, ensure_ascii=False, indent=2)})")

                result = execute_tool(
                    self._store, self._index,
                    session if has_session else _dummy_session(),
                    tc.function.name, arguments,
                )

                if self._verbose:
                    print(f"  ← {_format_result(result)}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

        if self._verbose:
            print(f"\n{'─'*50}")
            print(f"[安全兜底] 超过 {MAX_TOOL_ROUNDS} 轮，强制输出")
            print(f"{'─'*50}\n")
        response = self._client.chat.completions.create(
            model=self._model, messages=messages,
        )
        if has_session:
            session.commit_round(msg_name)
        return response.choices[0].message.content or ""


def _dummy_session() -> SessionMemory:
    """Fallback session when none provided (tests)."""
    from aad.embedder import Embedder
    return SessionMemory(Embedder(dim=4), dim=4)
