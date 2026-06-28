"""LLM Agent that orchestrates AAD tool calls via DeepSeek Chat API."""

import json
from typing import Any

from openai import OpenAI

from aad.store import AADStore
from aad.vector_index import VectorIndex
from aad.tools import TOOL_SCHEMAS, execute_tool

SYSTEM_PROMPT = """你是一个 AAD（关联激活扩散）知识图谱助手。
你的目标是通过探索关联知识图谱来回答用户问题。

工作流程：
1. 从用户输入中识别关键概念词元。
2. 对于你知道名称的概念，使用 `aad_lookup` 查找节点。
3. 检查返回的关联列表。对任何看起来相关的关联向量，
   调用 `aad_expand` 来寻找连接的节点。
4. 使用 `aad_get_content` 加载你发现节点的详细内容。
5. 当你收集到足够的上下文后，用自然语言综合回答。

重要规则：
- 当你已收集足够信息时，直接输出文本回答，不要调用工具。
- 当 `aad_lookup` 返回错误时，尝试替代名称或告知用户。
- 迭代探索图谱：查找揭示关联，每个关联可以展开找到更多节点。
- 综合回答用自然语言，不要留给用户原始 JSON 或未完成的工具调用。"""

MAX_TOOL_ROUNDS = 10  # 安全兜底，LLM 通常 2-4 轮自主停止


class AADAgent:
    """Orchestrates AAD tool calls via DeepSeek Chat API.

    Usage:
        agent = AADAgent(store, index, api_key="sk-...")
        answer = agent.run("黄仁勋是谁？")
    """

    def __init__(
        self,
        store: AADStore,
        index: VectorIndex,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
    ) -> None:
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is required")
        self._store = store
        self._index = index
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    def run(self, user_query: str) -> str:
        """Process a user query end-to-end. Returns the final answer string."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ]

        for _round in range(MAX_TOOL_ROUNDS):
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
            )

            msg = response.choices[0].message

            # LLM 自主决定：无 tool_calls → 已收集足够上下文，停止循环
            if not msg.tool_calls:
                return msg.content or ""

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

            # 执行所有工具调用，收集结果
            for tc in msg.tool_calls:
                try:
                    arguments = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    arguments = {}
                result = execute_tool(
                    self._store,
                    self._index,
                    tc.function.name,
                    arguments,
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })

        # 安全兜底：最后一轮不传 tools，强制输出文本回答
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        return response.choices[0].message.content or ""
