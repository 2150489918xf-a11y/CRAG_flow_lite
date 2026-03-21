# RAGFlow Lite

轻量化 RAG 系统，基于 [RAGFlow](https://github.com/infiniflow/ragflow) 核心检索架构构建。
**无需 Docker 25GB 全家桶**，只需一个 ES 实例即可运行完整的 RAG 检索服务。

## ✨ 核心功能

- **混合检索** — ES fulltext query_string + KNN 向量检索，融合权重可配
- **9 种智能分块** — naive / qa / laws / one / book / paper / presentation / table / manual
- **Parent-Child 层次分块** — 粗粒度父块用于上下文，细粒度子块用于精确召回
- **GraphRAG 图谱检索** — LLM 实体关系提取 → PageRank → 四路并行检索 → 融合
- **CRAG 纠错增强** — Correct / Incorrect / Ambiguous 三路动态路由，支持 Web 搜索兜底
- **BGE Reranker 精排** — 混合召回后二次排序
- **LLM 查询增强** — 同义词扩展 + 多语言翻译 + 关键词提取
- **DeepDoc 视觉引擎** — PDF 版面分析 + OCR + 表格结构识别
- **Agent Tool API** — OpenAI Function Calling 兼容，供外部 Agent 直接调用
- **知识库文件夹管理** — 虚拟路径嵌套，树形结构组织知识库
- **插件注册器** — 统一注册机制，存储/LLM/解析器/分块器均可热插拔

## 🏗️ 架构设计

```
┌─────────────────────────────────────────────────────┐
│  api/app.py  (入口壳 ~70 行)                         │
│  ├── routes/kb.py      知识库 CRUD + 文件夹管理       │
│  ├── routes/doc.py     文档上传/管理                   │
│  ├── routes/search.py  混合检索 + GraphRAG + CRAG     │
│  └── routes/tool.py    Agent Tool API (外部调用接口)   │
├─────────────────────────────────────────────────────┤
│  deps.py               共享依赖注入 (单例工厂)         │
│  models.py             Pydantic 请求/响应模型          │
├─────────────────────────────────────────────────────┤
│  common/registry.py    统一插件注册器                   │
│  common/perf.py        性能监控                        │
├─────────────────────────────────────────────────────┤
│  rag/llm/              LLM 抽象层 (ABC + Registry)    │
│  ├── base.py           BaseChatClient / BaseEmbedding │
│  │                     / BaseReranker (ABC + 工厂)    │
│  ├── chat.py           OpenAI 兼容 Chat 实现           │
│  ├── embedding.py      远程 Embedding 实现             │
│  └── reranker.py       远程 Reranker 实现              │
├─────────────────────────────────────────────────────┤
│  rag/utils/            存储抽象层 (ABC + Registry)     │
│  ├── doc_store_conn.py DocStoreConnection (18 抽象方法)│
│  └── es_conn.py        Elasticsearch 实现              │
├─────────────────────────────────────────────────────┤
│  rag/parser/           文档解析器 (ABC + Registry)     │
│  ├── base.py           BaseParser 抽象基类             │
│  └── pdf|docx|excel|…  8 种格式解析器                  │
├─────────────────────────────────────────────────────┤
│  rag/app/              分块引擎 (ABC + Registry)      │
│  ├── base_chunker.py   BaseChunker 抽象基类            │
│  ├── chunking.py       Router 工厂                    │
│  └── naive|qa|laws|…   9 种策略脚本                    │
├─────────────────────────────────────────────────────┤
│  rag/graph/            GraphRAG (提取/存储/检索)        │
│  rag/crag/             CRAG (评估/路由/提炼/Web搜索)    │
│  deepdoc/              视觉引擎 (版面分析/OCR/表格识别)  │
└─────────────────────────────────────────────────────┘
```

### 插件注册器架构

所有核心组件均通过 `common/registry.py` 统一管理，换后端只需实现新子类 + 改配置：

```python
# 新增后端只需 3 步：
# 1. 继承抽象基类
# 2. @registry.register("your_backend")
# 3. 修改 service_conf.yaml 的 backend 字段

@doc_store_registry.register("milvus")
class MilvusConnection(DocStoreConnection):
    ...  # 实现抽象方法即可，路由层零改动
```

| 注册器 | 管理对象 | 已注册后端 |
|--------|---------|-----------|
| `doc_store_registry` | 文档存储 | elasticsearch |
| `chat_registry` | Chat LLM | openai |
| `embedding_registry` | Embedding | openai |
| `reranker_registry` | Reranker | remote |
| `parser_registry` | 文档解析器 | pdf, docx, xlsx, md, txt, html, json, pptx |
| `chunker_registry` | 分块策略 | naive, qa, laws, one, … |

## 🚀 快速开始

### 1. 启动 Elasticsearch

```bash
docker-compose up -d
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置

编辑 `conf/service_conf.yaml`：

```yaml
embedding:
  api_key: "your-api-key"
  model_name: "BAAI/bge-m3"
  base_url: "https://api.siliconflow.cn/v1"

llm:
  api_key: "your-api-key"
  model_name: "deepseek-ai/DeepSeek-V3"
  base_url: "https://api.siliconflow.cn/v1"

# 可选：切换存储后端
doc_store:
  backend: elasticsearch  # 改为 milvus / qdrant 等
```

### 4. 构建索引

```bash
python scripts/build_index.py --kb_id my_kb --docs_dir ./data/documents/
```

### 5. 启动服务

```bash
python -m api.app
# 或
uvicorn api.app:app --host 0.0.0.0 --port 9380 --reload
```

## 📡 API 端点 (26 routes)

启动后访问 http://localhost:9380/docs 查看 Swagger 文档。

### 知识库管理

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/health` | GET | 健康检查（ES 状态 + 功能开关） |
| `/api/knowledgebase` | POST | 创建知识库（支持中文名 + 文件夹） |
| `/api/knowledgebase` | GET | 列出知识库（支持 folder 过滤） |
| `/api/knowledgebase/{kb_id}` | DELETE | 删除知识库 |
| `/api/knowledgebase/batch_delete` | POST | 批量删除知识库 |
| `/api/stats` | GET | 管道性能统计 (P50/P95/Avg) |
| `/api/stats/reset` | POST | 重置性能统计 |

### 文件夹管理

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/knowledgebase/folder` | POST | 创建文件夹（自动创建父级） |
| `/api/knowledgebase/folder` | DELETE | 删除空文件夹 |
| `/api/knowledgebase/move` | POST | 移动知识库到目标文件夹 |
| `/api/knowledgebase/tree` | GET | 返回文件夹树形结构 |

### 文档管理

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/documents/{kb_id}` | GET | 列出知识库文档 |
| `/api/chunks/{kb_id}` | GET | 查看分块内容（分页） |
| `/api/document/{kb_id}/{doc_name}` | DELETE | 删除指定文档 |
| `/api/document/upload` | POST | 上传并解析文档 |

### 检索服务

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/retrieval` | POST | 混合检索 + Reranker |
| `/api/graph_retrieval` | POST | GraphRAG + CRAG 增强检索 |

### Agent Tool API

供外部 Agent 系统调用的工具接口，纯检索工具，不做 LLM 生成。

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/tool/retrieve` | POST | 核心检索（fast/hybrid/deep 三种模式） |
| `/api/tool/list_kbs` | GET | 列出可用知识库（含文件夹信息） |
| `/api/tool/schema` | GET | 返回 OpenAI Function Calling JSON Schema |

**Agent 调用示例**：

```python
import requests

# 检索
resp = requests.post("http://localhost:9380/api/tool/retrieve", json={
    "query": "微软2024年投资了多少",
    "mode": "deep",          # fast | hybrid | deep
    "folder": "/财务",        # 可选：按文件夹过滤
})
data = resp.json()["data"]

# data["answer_context"]  → 拼好的上下文，直接塞进 LLM Prompt
# data["sources"]         → 溯源列表 [{id, content, doc_name, source_type}]
# data["metadata"]        → {mode, latency_ms, crag_score, ...}
```

## 📁 项目结构

```
RAGFlow_Lite/
├── api/                        # API 服务层
│   ├── app.py                 # FastAPI 入口壳
│   ├── deps.py                # 共享依赖注入
│   ├── models.py              # Pydantic 数据模型
│   ├── errors.py              # 统一错误处理
│   └── routes/                # 路由模块
│       ├── kb.py              # 知识库 + 文件夹管理
│       ├── doc.py             # 文档管理
│       ├── search.py          # 检索服务
│       └── tool.py            # Agent Tool API
├── common/                     # 公共基建
│   ├── registry.py            # 统一插件注册器
│   ├── perf.py                # 性能监控
│   └── log_config.py          # 日志配置
├── rag/                        # 核心 RAG 引擎
│   ├── llm/                   # LLM 抽象层 (ABC + Registry)
│   │   ├── base.py            # ABC 接口 + 注册器工厂
│   │   ├── chat.py            # Chat 客户端
│   │   ├── embedding.py       # Embedding 客户端
│   │   └── reranker.py        # Reranker 客户端
│   ├── nlp/                   # NLP 处理
│   │   ├── search.py          # 混合检索引擎
│   │   ├── query.py           # 查询扩展
│   │   ├── query_enhance.py   # LLM 查询增强
│   │   └── tokenizer.py       # jieba 分词器
│   ├── parser/                # 文档解析器 (ABC + Registry)
│   │   ├── base.py            # BaseParser 抽象基类
│   │   └── pdf|docx|excel|…   # 8 种格式解析器
│   ├── app/                   # 分块策略引擎 (ABC + Registry)
│   │   ├── base_chunker.py    # BaseChunker 抽象基类
│   │   ├── chunking.py        # Router 工厂
│   │   └── naive|qa|laws|…    # 9 种策略脚本
│   ├── graph/                 # GraphRAG
│   │   ├── extractor.py       # LLM 实体关系提取
│   │   ├── graph_store.py     # 图谱存储 + PageRank
│   │   └── graph_search.py    # 四路并行检索
│   ├── crag/                  # CRAG 纠错增强
│   │   ├── router.py          # 动态路由控制台
│   │   ├── evaluator.py       # 联合裁判器
│   │   ├── refiner.py         # 知识提炼器
│   │   └── web_search.py      # Web 搜索兜底
│   └── utils/                 # 存储抽象层 (ABC + Registry)
│       ├── doc_store_conn.py  # DocStoreConnection (18 抽象方法)
│       └── es_conn.py         # Elasticsearch 实现
├── deepdoc/                    # DeepDoc 视觉引擎
│   ├── parser/                # PDF/表格解析器
│   └── vision/                # OCR/版面分析
├── conf/                       # 配置文件
│   ├── service_conf.yaml      # 服务配置
│   ├── mapping.json           # ES 索引映射
│   └── folders.json           # 文件夹元数据
├── tests/                      # 测试套件 (86 tests)
├── scripts/                    # 离线工具
├── static/                     # 前端页面
└── docker-compose.yml          # ES Docker
```

## 🔧 可扩展性

| 扩展点 | 新增方式 | 需改动 |
|--------|---------|--------|
| 存储后端 (Milvus/Qdrant) | 继承 `DocStoreConnection` + 注册 | 0 行业务代码 |
| LLM 后端 (本地模型/Gemini) | 继承 `BaseChatClient` + 注册 | 0 行业务代码 |
| 文档解析器 (新格式) | 继承 `BaseParser` + 注册 | 0 行路由代码 |
| 分块策略 (新策略) | 继承 `BaseChunker` + 注册 | 0 行路由代码 |

## 📜 许可

基于 RAGFlow 开源项目构建，仅供学习研究使用。
