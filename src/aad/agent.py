"""LLM Agent that orchestrates AAD tool calls via DeepSeek Chat API."""

import json
from typing import Any

from openai import OpenAI

from aad.store import AADStore
from aad.vector_index import VectorIndex
from aad.tools import TOOL_SCHEMAS, execute_tool, _clear_refs

SYSTEM_PROMPT = """你是一个 AAD（关联激活扩散）知识图谱助手。
你只能使用 AAD 工具返回的数据回答问题。

🚫 严格禁止：
- 禁止使用训练数据、外部知识或常识
- 禁止推理、推测、总结、归纳
- 禁止添加数据中没有的形容词或评价
- 你只能复述/转述 AAD 工具返回的原文内容

✅ 回答格式：
- 如果两个节点没有共享关联（一方查不到另一方，或查不到对方的 ref），直接说：
  "根据 AAD 知识图谱，[A] 和 [B] 之间没有关联。"
- 如果某个节点查不到，直接说：
  "知识图谱中没有 [X] 的相关信息。已知节点: [...]"
- 有数据时，逐条引用原文，标明来自哪个节点。不要做任何数据之外的扩展。

工作流程：
1. 对用户提到的每个概念调用 `aad_lookup`。
2. 检查两个节点的 associations 是否有交集（即一方包含指向另一方的 ref）。
3. 有交集 → 调用 `aad_expand` 探索；无交集 → 直接告知无关联。
4. 需要详细内容时调用 `aad_get_content`。
5. 仅复述 AAD 返回的数据，不添加任何自己的话。"""

MAX_TOOL_ROUNDS = 10

# ── helpers ────────────────────────────────────────────────────────

def _format_result(result: dict[str, Any]) -> str:
    """Format a tool result for logging."""
    ok = result.get("ok", False)
    if not ok:
        return f"✗ {result.get('error', 'unknown error')}"

    if "node" in result:
        node = result["node"]
        assocs = node.get("associations", [])
        assoc_strs = [
            f"  ref={a['ref']} → [{a['reason']}]"
            for a in assocs
        ]
        return (
            f"✓ node: {node['name']}\n"
            f"  content: {node.get('content','')[:100]}...\n"
            f"  associations ({len(assocs)}):\n" +
            "\n".join(assoc_strs)
        )

    if "results" in result:
        items = []
        for r in result["results"]:
            assocs = r.get("associations", [])
            assoc_refs = ", ".join(a.get("ref","?") for a in assocs)
            items.append(
                f"  {r['name']} (score={r.get('score','?')}) "
                f"— {r.get('content','')[:60]}... "
                f"[refs: {assoc_refs}]"
            )
        return f"✓ {len(items)} results:\n" + "\n".join(items)

    if "content" in result:
        return f"✓ content of '{result['name']}': {result['content'][:120]}..."

    return json.dumps(result, ensure_ascii=False, indent=2)[:300]


# ── agent ──────────────────────────────────────────────────────────

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

    def run(self, user_query: str) -> str:
        _clear_refs()  # fresh ref table each query
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
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

            # 无 tool_calls → LLM 自主停止
            if not msg.tool_calls:
                if self._verbose:
                    print(f"\n{'─'*50}")
                    print(f"[轮次 {round_num}] LLM 决定停止，输出最终答案")
                    print(f"{'─'*50}\n")
                return msg.content or ""

            if self._verbose:
                print(f"\n{'─'*50}")
                print(f"[轮次 {round_num}] LLM 调用 {len(msg.tool_calls)} 个工具")

            # 追加 assistant 消息
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

            # 执行工具
            for tc in msg.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}

                if self._verbose:
                    args_str = json.dumps(arguments, ensure_ascii=False, indent=2)
                    print(f"  → {tc.function.name}({args_str})")

                result = execute_tool(
                    self._store, self._index,
                    tc.function.name, arguments,
                )

                if self._verbose:
                    print(f"  ← {_format_result(result)}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

        # 安全兜底
        if self._verbose:
            print(f"\n{'─'*50}")
            print(f"[安全兜底] 超过 {MAX_TOOL_ROUNDS} 轮，强制输出")
            print(f"{'─'*50}\n")
        response = self._client.chat.completions.create(
            model=self._model, messages=messages,
        )
        return response.choices[0].message.content or ""
