# AAD — Associative Activation Diffusion

基于向量的关联激活扩散知识图谱，带短期/长期双记忆系统和 LLM Agent 工具调用。

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
