# AAD — Associative Activation Diffusion

> 替换传统向量数据库：将存储、索引、推理统一为一张向量关联图谱，LLM 沿关联路径自主发散探索，大幅节省 Token 并消除幻觉。

## 动机：向量数据库的问题

当前 LLM 后端架构的标准做法是「Embedding → 向量数据库 → Top-K 检索 → 拼接进 Prompt」。

这套流程有三个根本缺陷：

**1. Token 浪费严重。** Top-K 检索返回的是一堆孤立文本片段，彼此没有关系。为了让 LLM 理解上下文，必须把所有片段全部塞进 Prompt。一个简单问题可能被灌入几千 Token 的检索结果，其中大部分与当前推理路径无关。

**2. 检索无状态，重复查询。** 多轮对话中，用户追问「它的产品呢」，系统无法利用上一轮已查过的节点，只能重新检索。同样的数据反复查、反复塞、反复消耗 Token。

**3. LLM 在检索结果之上叠加外部知识。** 当检索结果不够精确时，LLM 会自动用训练数据填补，产生无法追溯的幻觉。你无法知道哪句话来自数据库，哪句话是 LLM 编的。

## AAD 的思路：用关联图谱替代向量数据库

AAD 不把知识存成文档片段，而是存成**节点 + 向量关联**的有向图。存储即索引，关联即检索路径。

```
传统 RAG:
  问题 → Embedding → 向量库 Top-K → [片段A][片段B][片段C] → 拼进 Prompt → LLM 回答
  问题：片段之间没有关系，LLM 被动接收，Token 膨胀

AAD:
  问题 → LLM 提取词元 → aad_lookup("黄仁勋")
       → 节点返回关联: [创立了 NVIDIA] [领导了 GPU 发明]
       → LLM 选择相关关联 → aad_expand(NVIDIA向量) → 命中 NVIDIA
       → NVIDIA 节点返回关联: [CUDA] [GeForce] [数据中心]
       → LLM 认为信息足够，停止探索，输出答案
  关键：每次只加载一个节点，LLM 自主决定下一步，Token 按需消耗
```

**核心差异**：

| | 向量数据库 + RAG | AAD 关联图谱 |
|---|---|---|
| 存储模型 | 文档片段，无关联 | 节点 + 向量关联，有向图 |
| 检索方式 | Top-K 相似度，一次性返回 | LLM 沿关联逐步发散，按需探索 |
| Token 消耗 | 每次塞入全部检索结果 | 每步只加载当前节点，步进式消耗 |
| 多轮对话 | 无状态，重复检索 | 短期记忆镜像已加载节点，直接命中 |
| 可追溯性 | 不知道 LLM 哪句话来自数据 | 每个回答都能回溯到具体节点 |
| 幻觉控制 | LLM 自由发挥 | 系统提示词禁止使用外部知识 |

## 长远目标

所有 LLM 共用一个持续增长的知识图谱作为「大脑」。推理时不需要向量库检索，直接沿图谱关联发散——推理速度受限于关联路径长度而非数据库规模。当前 AAD 1.0 是这个方向的极简原型：17 节点，3 个工具，不到 600 行核心代码，但架构完整。

## 快速开始

```bash
# 1. 安装
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. 配置
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY

# 3. 运行
python -m aad.cli
# 首次运行自动生成 17 个种子节点，之后进入交互问答
```

## 架构

```
用户输入 → CLI → AADAgent.run()
  ├── 短期记忆 (SessionMemory) — 会话级，/quit 销毁
  │     ├── msg_N 消息节点（线性链）
  │     └── 长期记忆镜像（自动拷贝，内容可改写）
  │
  └── DeepSeek Chat API（工具调用循环，LLM 自主停止）
        ├── aad_lookup(name)           → 短期优先 → 长期 → 自动镜像
        ├── aad_expand(ref, top_k)     → 双索引搜索（短期 + 长期）
        └── aad_get_content(name)      → 获取节点完整内容
```

| 组件 | 说明 |
|------|------|
| **Agent LLM** | DeepSeek Chat API，工具调用原生支持 OpenAI function-calling 格式 |
| **嵌入** | 本地确定性 hash（SHA-256 → 归一化向量），零外部依赖 |
| **长期记忆** | `AADStore`：内存 HashMap + JSONL 持久化，跨会话保留 |
| **短期记忆** | `SessionMemory`：会话级知识图谱，自动镜像 + 推理关联，/quit 丢弃 |
| **向量索引** | FAISS `IndexFlatIP`（长短各一个独立索引） |
| **SDK** | `openai` Python SDK（`base_url` 指向 `api.deepseek.com/v1`） |

## 种子数据

17 个节点，含 GPU 产业链、竞争对手和干扰项：

```
核心三角: 黄仁勋 ←→ NVIDIA ←→ GPU
GPU 链:   GPU → 图形渲染 / 深度学习 / GeForce / RTX / 比特币
NVIDIA 链: NVIDIA → CUDA / 数据中心 / GeForce / RTX / AMD
关联延伸: CUDA → 深度学习 → 人工智能
          图形渲染 → 光线追踪 ← RTX
          数据中心 → 人工智能
竞争对手: AMD（关联到 NVIDIA 和 GPU）
干扰项:   苹果 / 特斯拉 / Python / 量子计算（零关联）
```

## CLI 命令

| 命令 | 功能 |
|------|------|
| `> 你的问题` | 输入查询，Agent 探索图谱并回答 |
| `/nodes` | 列出所有长期记忆节点 |
| `/session` | 查看短期记忆状态（消息数、镜像节点） |
| `/quit`, `/q` | 退出（短期记忆自动清除） |
| `--reseed` | 启动时强制重生种子数据 |

## 运行测试

```bash
python -m pytest -v
```

## 项目结构

```
src/aad/
├── config.py         # 配置（.env → pydantic-settings）
├── errors.py         # 异常层级
├── models.py         # Node, Association (Pydantic v2)
├── embedder.py       # 确定性 hash 嵌入（原型验证）
├── store.py          # 长期记忆：HashMap + JSONL
├── vector_index.py   # FAISS IndexFlatIP 封装
├── session.py        # 短期记忆：会话图谱 + 消息链 + 自动镜像
├── tools.py          # 3 个工具 Schema + 实现 + 调度器
├── agent.py          # DeepSeek Chat 工具调用循环
├── seed.py           # 种子数据（17 节点）
└── cli.py            # 交互式 CLI
```
