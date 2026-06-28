import pytest
import json
import hashlib
from unittest.mock import patch, MagicMock

from aad.agent import AADAgent, SYSTEM_PROMPT_BASE, MAX_TOOL_ROUNDS
from aad.session import SessionMemory
from aad.models import Node, Association


def _fake_embedder(dim=4):
    def fake_embed(text):
        h = hashlib.sha256(text.encode()).digest()[:dim]
        return [float(b) / 255.0 for b in h]
    emb = MagicMock()
    emb.embed.side_effect = fake_embed
    emb.embed_batch.side_effect = lambda texts: [fake_embed(t) for t in texts]
    return emb


@pytest.fixture
def mock_store():
    store = MagicMock()
    store.list_names.return_value = ["GPU", "NVIDIA", "黄仁勋"]
    store.get.return_value = Node(
        name="GPU", content="Graphics Processing Unit",
        vector=[0.1, 0.2, 0.3, 0.4],
        associations=[Association(vector=[0.4, 0.5, 0.6, 0.7], reason="made by")],
    )
    store.get_content.return_value = "Graphics Processing Unit"
    return store


@pytest.fixture
def mock_index():
    index = MagicMock()
    index.search.return_value = [("GPU", 0.99), ("NVIDIA", 0.85)]
    return index


@pytest.fixture
def session():
    return SessionMemory(_fake_embedder(4), dim=4)


@pytest.fixture
def agent(mock_store, mock_index):
    return AADAgent(mock_store, mock_index, api_key="sk-test", verbose=False)


def _make_text_response(text):
    msg = MagicMock()
    msg.content = text
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    response = MagicMock()
    response.choices = [choice]
    return response


def _make_tool_response(tool_calls):
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

    def test_run_direct_answer_no_tools(self, agent, session):
        with patch.object(agent._client.chat.completions, "create",
                          return_value=_make_text_response("答案是 NVIDIA。")):
            result = agent.run("黄仁勋的 GPU 公司叫什么？", session=session)
            assert "NVIDIA" in result

    def test_run_single_tool_call_then_answer(self, agent, session):
        with patch.object(agent._client.chat.completions, "create",
                          side_effect=[
                              _make_tool_response([("call_1", "aad_lookup", json.dumps({"name": "黄仁勋"}))]),
                              _make_text_response("黄仁勋是 NVIDIA 的创始人。"),
                          ]):
            result = agent.run("黄仁勋是谁？", session=session)
            assert "NVIDIA" in result

    def test_run_creates_message_node(self, agent, session):
        """Session gets a msg_N node."""
        with patch.object(agent._client.chat.completions, "create",
                          return_value=_make_text_response("OK")):
            agent.run("测试", session=session)
        assert session.is_visited("msg_1")
        assert session.get_node("msg_1").content == "测试"

    def test_run_session_summary_in_prompt(self, agent, session):
        """System prompt includes session summary."""
        captured = []

        def capture(*args, **kwargs):
            captured.append(kwargs.get("messages", []))
            return _make_text_response("OK")

        with patch.object(agent._client.chat.completions, "create", side_effect=capture):
            agent.run("hello", session=session)

        system = captured[0][0]["content"]
        assert "[短期记忆状态]" in system
        assert "消息节点" in system

    def test_run_max_rounds_safety_net(self, agent, session):
        tool_responses = [
            _make_tool_response([("c1", "aad_lookup", json.dumps({"name": "GPU"}))])
        ] * MAX_TOOL_ROUNDS
        final = _make_text_response("已收集足够上下文。")
        with patch.object(agent._client.chat.completions, "create",
                          side_effect=tool_responses + [final]):
            result = agent.run("Query", session=session)
            assert "已收集足够上下文" in result

    def test_run_without_session_still_works(self, agent):
        """Backward compatible: no session passed."""
        with patch.object(agent._client.chat.completions, "create",
                          return_value=_make_text_response("OK")):
            result = agent.run("测试")
            assert "OK" in result
