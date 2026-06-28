import pytest
import hashlib
from unittest.mock import MagicMock

from aad.session import SessionMemory
from aad.models import Node, Association
from aad.vector_index import VectorIndex


def _fake_embedder(dim=4):
    def fake_embed(text):
        h = hashlib.sha256(text.encode()).digest()[:dim]
        return [float(b) / 255.0 for b in h]
    emb = MagicMock()
    emb.embed.side_effect = fake_embed
    emb.embed_batch.side_effect = lambda texts: [fake_embed(t) for t in texts]
    return emb


@pytest.fixture
def session():
    return SessionMemory(_fake_embedder(4), dim=4)


class TestSessionMemory:
    def test_add_message_creates_node(self, session):
        name = session.add_message("你好")
        assert name == "msg_1"
        assert session.is_visited("msg_1")
        node = session.get_node("msg_1")
        assert node.content == "你好"
        assert len(node.vector) == 4

    def test_message_linear_chain(self, session):
        session.add_message("第一条")
        session.add_message("第二条")
        msg2 = session.get_node("msg_2")
        # msg_2 should have an association to msg_1
        chain_assoc = [a for a in msg2.associations if a.reason == "对话顺序"]
        assert len(chain_assoc) == 1
        # Check the vector matches msg_1
        msg1 = session.get_node("msg_1")
        assert chain_assoc[0].vector == msg1.vector

    def test_mirror_longterm(self, session):
        lt_node = Node(
            name="GPU",
            content="Graphics Processing Unit",
            vector=[1.0, 2.0, 3.0, 4.0],
            associations=[Association(vector=[0.1, 0.2, 0.3, 0.4], reason="made by")],
        )
        mirror = session.mirror_longterm(lt_node)
        assert mirror.name == "GPU"
        assert mirror.content == lt_node.content
        assert mirror.vector == lt_node.vector
        # Associations ARE copied (shallow), rewritable later
        assert len(mirror.associations) == 1
        assert mirror.associations[0].reason == "made by"
        assert session.is_visited("GPU")
        assert session.mirrored_count == 1

    def test_mirror_idempotent(self, session):
        lt = Node(name="X", content="c", vector=[1, 2, 3, 4])
        m1 = session.mirror_longterm(lt)
        m2 = session.mirror_longterm(lt)
        assert m1 is m2
        assert session.mirrored_count == 1

    def test_link_message_to_node(self, session):
        session.add_message("测试问题")
        lt = Node(name="NVIDIA", content="NVIDIA Corp", vector=[0, 1, 0, 0])
        session.mirror_longterm(lt)
        session.link_message_to_node("msg_1", "NVIDIA", "推理引用")
        msg = session.get_node("msg_1")
        refs = [a for a in msg.associations if a.reason == "推理引用"]
        assert len(refs) == 1
        assert refs[0].vector == lt.vector

    def test_link_message_no_duplicate(self, session):
        session.add_message("q")
        lt = Node(name="A", content="a", vector=[1, 0, 0, 0])
        session.mirror_longterm(lt)
        session.link_message_to_node("msg_1", "A", "推理引用")
        session.link_message_to_node("msg_1", "A", "推理引用")
        # Should not duplicate
        msg = session.get_node("msg_1")
        refs = [a for a in msg.associations if a.reason == "推理引用"]
        assert len(refs) == 1

    def test_track_and_commit_round(self, session):
        session.add_message("q1")
        session.begin_round()
        lt = Node(name="B", content="b", vector=[0, 0, 1, 0])
        session.mirror_longterm(lt)
        session.track_node("B")
        session.commit_round("msg_1")
        msg = session.get_node("msg_1")
        refs = [a for a in msg.associations if a.reason == "推理引用"]
        assert len(refs) == 1
        assert refs[0].vector == lt.vector

    def test_summary_includes_mirrored(self, session):
        lt = Node(name="GPU", content="gpu", vector=[1, 0, 0, 0])
        session.mirror_longterm(lt)
        s = session.summary()
        assert "GPU" in s
        assert "消息" in s

    def test_get_index_returns_vector_index(self, session):
        idx = session.get_index()
        assert isinstance(idx, VectorIndex)

    def test_clear_resets_everything(self, session):
        session.add_message("hello")
        lt = Node(name="X", content="x", vector=[1, 2, 3, 4])
        session.mirror_longterm(lt)
        session.clear()
        assert session.node_count == 0
        assert session.mirrored_count == 0
        assert not session.is_visited("msg_1")
        assert not session.is_visited("X")
