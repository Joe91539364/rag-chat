# 群聊 RAG 助手（RagChat）

> 面向班级管理的本地 AI 助手：把散落在微信群的历史消息做向量化检索，支持语义问答、多后端 LLM 生成与可量化的检索评测。

一个完整可跑的 **RAG（Retrieval-Augmented Generation，检索增强生成）** 项目，从词袋基线逐步升级到真实语义嵌入 + 大模型生成，并用 recall@k 量化证明了"为什么 RAG 要用预训练语义模型"。

## 项目亮点

- **端到端可跑**：数据 → 向量化 → 语义检索 → LLM 生成，一条命令跑通
- **有评测数据**：用 recall@k 量化对比词袋基线 vs bge-m3，用数字证明技术选型
- **多后端可切换**：智谱 / 硅基流动 / DeepSeek，换模型只改一个参数
- **免费可用**：默认走智谱 GLM-4-Flash（永久免费），零成本跑通
- **本地推理**：bge-m3 嵌入在本地 GPU（RTX 4060）上运行

## 核心链路

```
加载数据 → 分块 → 向量化 → 建索引 → 语义检索 → 生成答案
```

## 技术栈

| 模块 | 当前实现 | 升级方向 |
|---|---|---|
| 嵌入 | bge-m3 句子嵌入（本地 GPU/CPU 跑） | 支持词袋基线做对照 |
| 检索 | 余弦相似度（手写 NumPy） + Chroma 向量库 | pgvector / Milvus |
| 生成 | 多后端 OpenAI 兼容接口（智谱免费） | 其它 LLM / 本地模型 |
| 评测 | recall@k（eval.py） | 回答质量评估 |
| 记忆 | 分层摘要 + 长期记忆（memory.py） | 向量化记忆检索 |
| 工具 | 待办 / 提醒 / 日程（agent.py） | 更多工具（邮件/日历/投票） |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 下载嵌入模型（首次，约 2.2GB）

```bash
python download_model.py
```

> 已内置国内网络处理：走 hf-mirror.com 镜像、禁用 xet、用 curl 正确跟随重定向，解决 Windows 下 huggingface_hub 下载空文件的坑。模型下载到 `models/bge-m3/`。

### 3. 检索问答

```bash
# bge-m3 嵌入（默认，语义更强）
python rag.py "体测是几号"

# 词袋基线（零依赖，秒出，用于对照）
python rag.py "体测是几号" --embedder bow

# Chroma 向量库检索（专业引擎，数据量大时更高效）
python rag.py "体测是几号" --retriever chroma
```

> 检索后端默认 `numpy`（手写余弦），用 `--retriever chroma` 切换到 Chroma 向量库。两种后端检索结果一致，Chroma 适合消息量大、需要持久化索引的场景（索引存在 `chroma_db/`）。

### 4. 生成式回答（接 LLM，支持多后端切换）

```bash
# 查看所有后端及配置状态
python generate.py --list

# 用默认后端（智谱免费）回答
python generate.py "体测是几号"

# 切换后端
python generate.py "体测是几号" --backend siliconflow
python generate.py "体测是几号" --backend deepseek

# 调整检索条数（默认 5，答案可能排在第 4、5 位）
python generate.py "体测是几号" --top-k 8
```

**支持的免费后端**（复制 `config.example.json` 为 `config.json` 并填入 api_key）：

| 后端 | 模型 | 费用 | 注册 |
|---|---|---|---|
| 智谱 zhipu | glm-4-flash | 永久免费 | open.bigmodel.cn |
| 硅基流动 siliconflow | DeepSeek-V3 等 | 送免费额度 | cloud.siliconflow.cn |
| DeepSeek 官方 | deepseek-chat | 低价 | platform.deepseek.com |

> 未配置任何 key 时自动降级为「检索拼接」，不报错。三个后端都兼容 OpenAI 格式，换模型只需改 `--backend` 或 `config.json` 里的 `default_backend`。

### 5. 评测检索效果

```bash
python eval.py --embedder bge             # bge-m3 的 recall@k（numpy 后端）
python eval.py --embedder bow             # 词袋基线对照
python eval.py --retriever chroma         # 用 Chroma 后端评测
```

### 6. 分层记忆（会话摘要 + 长期记忆沉淀）

```bash
# 从全部消息生成会话摘要 + 抽取长期记忆事实（沉淀到 data/memory_store.json）
python memory.py

# 关键词检索长期记忆
python memory.py --recall "体测"
```

> 设计是「记忆复利」：短期=原始消息，中期=LLM 压缩的会话摘要，长期=从摘要中抽取的自包含事实。用一次，沉淀一批，越用越懂这个班。

### 7. Agent 工具调用（待办 / 提醒 / 日程）

```bash
# 设提醒
python agent.py "提醒我周五下午3点开班会"

# 加待办
python agent.py "帮我记一下，明天要交高数作业"

# 查日程（汇总待办 + 提醒）
python agent.py "我最近有什么待办？"
```

> 用 LLM 判断意图并抽取参数，落到本地 JSON（`data/agent_store.json`）。未配置 key 时降级为关键词规则，不报错。

## 端到端验证（已跑通）

用智谱 GLM-4-Flash 免费模型实测，检索 + 生成闭环完整：

| 问题 | LLM 回答 |
|---|---|
| 体测是几号 | 体测是下个月10号 |
| 奖学金要什么绩点才能评 | 奖学金评定的成绩门槛是绩点3.0以上 |
| 宿舍晚上几点关门 | 宿舍晚上11点关门 |

> 实践发现：语义检索中，答案可能排在第 4、5 位（问题与"相似问题"的距离有时比与"答案"的距离更近），因此默认 top-k=5。这是 RAG 实践中真实存在的现象。

## 评测结果（136 条消息 + 20 个问题）

| 嵌入方式 | 向量维度 | recall@1 | recall@2 | recall@3 |
|---|---|---|---|---|
| 词袋 + TF-IDF | ~2000 | 0.200 | 0.550 | 0.750 |
| bge-m3 | 1024 | **0.450** | **0.950** | **1.000** |

**关键结论**：语料扩大到 136 条、问题语义化后，词袋的 recall@1 从 0.917 暴跌到 0.200（大量语义改写问题答错），而 bge-m3 保持 0.450、recall@2 就接近满分。**语义理解优势在真实规模语料上才真正体现**——这正是 RAG 要换真实嵌入模型的根本原因。

## 数据扩充

```bash
python gen_data.py   # 生成 136 条贴近真实班级群聊的消息
```

## 目录结构

```
rag-prototype/
├── rag.py              # 核心：加载/索引/检索/生成
├── generate.py         # LLM 生成层（多后端可切换 + 通用 call_llm）
├── memory.py           # 分层记忆：会话摘要 + 长期记忆沉淀
├── agent.py            # Agent 工具调用：待办/提醒/日程
├── eval.py             # 评测：recall@k
├── gen_data.py         # 生成扩充数据集（136 条）
├── download_model.py   # 下载 bge-m3 模型（含国内网络处理）
├── config.example.json # LLM 后端配置模板
├── requirements.txt
├── .gitignore
├── data/
│   ├── messages.json    # 群聊消息样本（136 条）
│   ├── memory_store.json# 分层记忆（摘要 + 长期记忆）
│   └── agent_store.json # Agent 工具数据（待办/提醒）
└── README.md
```

## 下一步计划

1. ✅ 接入真实嵌入模型（sentence-transformers + bge-m3）
2. ✅ 接入 LLM API（多后端可切换）
3. ✅ 增加评测集，量化 recall@k
4. ✅ 换用向量数据库，支持大规模消息（Chroma，`--retriever chroma`）
5. ✅ 分层摘要 + 长期记忆（记忆复利，`memory.py`）
6. ✅ Agent 化：工具调用（日历/待办/提醒，`agent.py`）

## 许可证

MIT License
