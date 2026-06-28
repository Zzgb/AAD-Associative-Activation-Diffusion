"""Seed data factory for the AAD knowledge graph.

Creates the demo chain: 黄仁勋 (Jensen Huang) → GPU → NVIDIA
with bidirectional associations.
"""

from aad.models import Node, Association
from aad.embedder import Embedder
from aad.store import AADStore


def create_seed_nodes(embedder: Embedder) -> list[Node]:
    """Create the seed nodes with embeddings and associations.

    Returns three nodes forming a bidirectional association chain:
      - 黄仁勋 (Jensen Huang), founder and CEO of NVIDIA
      - GPU (Graphics Processing Unit), invented by NVIDIA
      - NVIDIA, the company

    All embeddings are generated via the provided Embedder instance.
    """
    nodes_data = [
        {
            "name": "黄仁勋",
            "content": (
                "黄仁勋 (Jensen Huang) 是 NVIDIA 公司的联合创始人、总裁兼首席执行官。"
                "他于 1963 年出生于台湾，1993 年联合创立了 NVIDIA，"
                "带领公司成为 GPU 计算和 AI 加速领域的主导力量。"
                "他以标志性的皮夹克和在加速计算领域的远见领导力而闻名。"
            ),
        },
        {
            "name": "GPU",
            "content": (
                "GPU（图形处理单元）是一种最初为渲染图形而设计的专用处理器。"
                "现代 GPU 是高度并行的处理器，用于通用计算（GPGPU），"
                "包括 AI 训练、科学模拟和加密货币挖矿。"
                "NVIDIA 于 1999 年推出了 GeForce 256，发明了 GPU。"
            ),
        },
        {
            "name": "NVIDIA",
            "content": (
                "NVIDIA 公司是一家美国跨国技术公司，由黄仁勋（Jensen Huang）、"
                "Chris Malachowsky 和 Curtis Priem 于 1993 年创立。"
                "总部位于加利福尼亚州圣克拉拉，NVIDIA 设计用于游戏、"
                "专业可视化、数据中心和 AI 的图形处理单元（GPU）。"
                "该公司的 CUDA 平台已成为 GPU 加速计算的事实标准。"
            ),
        },
    ]

    # Generate embeddings from name + content
    texts = [d["name"] + " " + d["content"] for d in nodes_data]
    vectors = embedder.embed_batch(texts)

    # Build nodes without associations first
    nodes: dict[str, Node] = {}
    for data, vec in zip(nodes_data, vectors):
        nodes[data["name"]] = Node(
            name=data["name"],
            content=data["content"],
            vector=vec,
        )

    # Add bidirectional associations

    # 黄仁勋 <-> NVIDIA (founded / founded by)
    nodes["黄仁勋"].associations.append(
        Association(
            vector=nodes["NVIDIA"].vector,
            reason="联合创立了 NVIDIA（1993 年）",
        )
    )
    nodes["NVIDIA"].associations.append(
        Association(
            vector=nodes["黄仁勋"].vector,
            reason="由黄仁勋（Jensen Huang）联合创立",
        )
    )

    # NVIDIA <-> GPU (invented / invented by)
    nodes["NVIDIA"].associations.append(
        Association(
            vector=nodes["GPU"].vector,
            reason="发明了 GPU（GeForce 256，1999 年）",
        )
    )
    nodes["GPU"].associations.append(
        Association(
            vector=nodes["NVIDIA"].vector,
            reason="GPU 由 NVIDIA 发明",
        )
    )

    # 黄仁勋 <-> GPU (led invention of)
    nodes["黄仁勋"].associations.append(
        Association(
            vector=nodes["GPU"].vector,
            reason="领导了发明 GPU 的公司",
        )
    )
    nodes["GPU"].associations.append(
        Association(
            vector=nodes["黄仁勋"].vector,
            reason="GPU 由黄仁勋领导下的 NVIDIA 发明",
        )
    )

    return list(nodes.values())


def seed_store(store: AADStore, embedder: Embedder) -> int:
    """Populate an AADStore with seed nodes. Returns count of nodes added."""
    nodes = create_seed_nodes(embedder)
    for node in nodes:
        store.put(node)
    return len(nodes)
