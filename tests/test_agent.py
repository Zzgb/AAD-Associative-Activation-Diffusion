import pytest
import json
from unittest.mock import patch, MagicMock

from aad.agent import AADAgent, SYSTEM_PROMPT, MAX_TOOL_ROUNDS
from aad.models import Node, Association


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.list_names.return_value = ["GPU", "NVIDIA", "黄仁勋"]
    # Return a real Node so model_dump() produces JSON-serializable data
    store.get.return_value = Node(
        name="GPU",
        content="Graphics Processing Unit",
        vector=[0.1, 0.2, 0.3],
        associations=[Association(vector=[0.4, 0.5], reason="made by")],
    )
    store.get_content.return_value = "Graphics Processing Unit"
    return store


@pytest.fixture
def mock_index():
    index = MagicMock()
    index.search.return_value = [("GPU", 0.99), ("NVIDIA", 0.85)]
    return index


@pytest.fixture
def agent(mock_store, mock_index):
    return AADAgent(mock_store, mock_index, api_key="sk-test", verbose=False)


def _make_text_response(text: str):
    """Helper: create a mock chat completion with text content only."""
    msg = MagicMock()
    msg.content = text
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_tool_response(tool_calls: list[tuple[str, str, str]]):
    """Helper: create a mock chat completion with tool calls.

    Each tuple: (id, name, arguments_json_string)
    """
    tcs = []
    for tc_id, name, args in tool_calls:
        tc = MagicMock()
        tc.id = tc_id
        tc.function.name = name
        tc.function.arguments = args
        tcs.append(tc)
    msg = MagicMock()
    msg.content = None
    msg.tool_calls = tcs
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


class TestAADAgent:
    def test_init_requires_api_key(self, mock_store, mock_index):
        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
            AADAgent(mock_store, mock_index, api_key="")

    def test_init_stores_dependencies(self, mock_store, mock_index):
        agent = AADAgent(
            mock_store, mock_index, api_key="sk-test",
            base_url="https://api.deepseek.com/v1", model="deepseek-chat"
        )
        assert agent._store is mock_store
        assert agent._index is mock_index
        assert agent._model == "deepseek-chat"

    def test_run_direct_answer_no_tools(self, agent):
        """LLM 直接返回文本，无工具调用 → 立即停止返回答案."""
        with patch.object(
            agent._client.chat.completions, "create",
            return_value=_make_text_response("答案是 NVIDIA。"),
        ):
            result = agent.run("黄仁勋的 GPU 公司叫什么？")
            assert "NVIDIA" in result

    def test_run_single_tool_call_then_answer(self, agent):
        """LLM 先调用一个工具，然后返回文本 → 正常停止."""
        with patch.object(
            agent._client.chat.completions, "create",
            side_effect=[
                _make_tool_response([
                    ("call_1", "aad_lookup", json.dumps({"name": "黄仁勋"}))
                ]),
                _make_text_response("黄仁勋是 NVIDIA 的创始人。"),
            ],
        ):
            result = agent.run("黄仁勋是谁？")
            assert "NVIDIA" in result

    def test_run_multi_tool_call_then_answer(self, agent):
        """LLM 调用两个工具后返回文本 → 正常停止."""
        with patch.object(
            agent._client.chat.completions, "create",
            side_effect=[
                _make_tool_response([
                    ("call_1", "aad_lookup", json.dumps({"name": "黄仁勋"})),
                    ("call_2", "aad_lookup", json.dumps({"name": "GPU"})),
                ]),
                _make_text_response("黄仁勋创立了 NVIDIA，发明了 GPU。"),
            ],
        ):
            result = agent.run("黄仁勋和 GPU 的关系？")
            assert "NVIDIA" in result

    def test_run_max_rounds_safety_net(self, agent):
        """超过 MAX_TOOL_ROUNDS 时强制输出最终回答."""
        # 每次响应都包含 tool_calls（模拟 LLM 一直调用工具）
        tool_responses = [
            _make_tool_response([("call_1", "aad_lookup", json.dumps({"name": "GPU"}))])
        ] * MAX_TOOL_ROUNDS

        final_text = _make_text_response("已收集足够上下文。")
        side_effects = tool_responses + [final_text]

        with patch.object(
            agent._client.chat.completions, "create",
            side_effect=side_effects,
        ):
            result = agent.run("Query")
            assert "已收集足够上下文" in result

    def test_system_prompt_included(self, agent):
        """验证系统提示词被包含在消息中."""
        captured_messages = []

        def capture(*args, **kwargs):
            captured_messages.append(kwargs.get("messages", []))
            return _make_text_response("OK")

        with patch.object(agent._client.chat.completions, "create", side_effect=capture):
            agent.run("test")

        first_message = captured_messages[0]
        assert first_message[0]["role"] == "system"
        assert first_message[0]["content"] == SYSTEM_PROMPT
        assert first_message[1]["role"] == "user"
        assert first_message[1]["content"] == "test"

    def test_tool_result_json_serializable(self, agent):
        """工具结果可以正确序列化并通过 tool 消息传递."""
        with patch.object(
            agent._client.chat.completions, "create",
            side_effect=[
                _make_tool_response([
                    ("call_1", "aad_lookup", json.dumps({"name": "GPU"}))
                ]),
                _make_text_response("GPU 是图形处理器。"),
            ],
        ):
            result = agent.run("什么是 GPU？")
            assert "GPU" in result
