# AAD — Associative Activation Diffusion

基于向量的关联记忆知识图谱，带 LLM Agent 工具调用。

## 快速开始

```bash
# 1. 创建虚拟环境并安装
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY

# 3. 运行 Agent
python -m aad.cli
```

首次运行时，会自动调用 DeepSeek 嵌入 API 生成种子数据（3 个节点：黄仁勋、GPU、NVIDIA），之后进入交互式问答。

## 架构

```
用户输入 → CLI → AADAgent.run()
  └── DeepSeek Chat API（工具调用循环，LLM 自主停止）
        ├── aad_lookup(name)           → HashMap<name, Node>
        ├── aad_expand(vector, top_k)  → FAISS IndexFlatIP ANN 搜索
        └── aad_get_content(name)      → Node.content 文本
      → LLM 认为上下文足够，停止调工具，输出最终答案
```

**存储**：内存 `dict[str, Node]` + JSONL 文件持久化  
**向量索引**：FAISS `IndexFlatIP`（内积 ≈ 余弦相似度）  
**嵌入**：DeepSeek 嵌入 API  
**Agent LLM**：DeepSeek Chat API，工具调用原生支持 OpenAI function-calling 格式  
**SDK**：统一使用 `openai` Python SDK（`base_url` 指向 `api.deepseek.com/v1`）

## 种子数据

三个节点双向全连接：

```
黄仁勋 ←→ NVIDIA（联合创立 / 由黄仁勋联合创立）
NVIDIA ←→ GPU（发明 / 由 NVIDIA 发明）
黄仁勋 ←→ GPU（领导发明 / 在黄仁勋领导下发明）
```

共 6 条关联边，嵌入由 `name + content` 拼接通过 DeepSeek 生成。

## 运行测试

```bash
python -m pytest -v
```

## CLI 命令

| 命令 | 功能 |
|------|------|
| `> 你的问题` | 输入查询，Agent 自动探索图谱并回答 |
| `/nodes` | 列出所有已知节点名称 |
| `/quit` | 退出 |

## 项目结构

```
src/aad/
├── config.py         # 配置（.env → pydantic-settings）
├── errors.py         # 异常层级
├── models.py         # Node, Association (Pydantic v2)
├── embedder.py       # DeepSeek 嵌入客户端
├── store.py          # HashMap + JSONL 存储
├── vector_index.py   # FAISS 封装
├── tools.py          # 3 个工具 Schema + 实现 + 调度器
├── agent.py          # DeepSeek Chat 工具调用循环
├── seed.py           # 种子数据工厂
└── cli.py            # 交互式 CLI
```
