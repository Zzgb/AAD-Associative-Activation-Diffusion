"""LLM Agent that orchestrates AAD tool calls via DeepSeek Chat API."""

import json
from typing import Any

from openai import OpenAI

from aad.store import AADStore
from aad.vector_index import VectorIndex
from aad.session import SessionMemory
from aad.tools import TOOL_SCHEMAS, execute_tool, _clear_refs

SYSTEM_PROMPT_BASE = """你是 AAD 知识图谱助手。你只能在图谱中探索，不能调用训练数据。

⛔ 致命规则（违反 = 错误回答）：
1. 你输出的每一句话，必须来自 AAD 工具返回的原文。逐字引用。
2. 工具没返回的内容 = 不存在。不能说"值得注意的是""值得一提"等延伸。
3. 禁止举例、禁止类比、禁止补充细节。工具返回什么就只说什么。
4. 如果两个节点查不到关联 → "AAD 图谱中无关联"，不加任何解释。

举个反例——以下回答是错的：
  "NVIDIA 旗下有 Tesla V100、A100 等产品"
  → 错：节点只写了"Tesla（数据中心）"，V100/A100 是编的。

✅ 正确做法：只复述节点原文。
  → NVIDIA 节点原文: "旗下拥有 GeForce、RTX、Quadro、Tesla（数据中心）等产品线"

🔀 遍历工具
  aad_lookup(name)       — 查节点
  aad_expand(ref)        — 沿一条关联发散
  aad_get_content(name)  — 获取完整内容
  aad_trace(from, to)    — 双向BFS最短路径

原则：
- 多个概念同时 lookup，不等
- 读关联 reason 判断方向，不相关的 skip
- 找路径用双向发散或 aad_trace
- 死胡同（无关联）立即终止
- 不用穷举，够了就停"""

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
