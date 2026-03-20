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

## 🏗️ 架构设计

```
┌─────────────────────────────────────────────────────┐
│  api/app.py  (入口壳 ~60 行)                         │
│  ├── routes/kb.py      知识库 CRUD                   │
│  ├── routes/doc.py     文档上传/管理                   │
│  └── routes/search.py  混合检索 + GraphRAG + CRAG     │
├─────────────────────────────────────────────────────┤
│  deps.py               共享依赖注入 (单例工厂)         │
│  models.py             Pydantic 请求/响应模型          │
├─────────────────────────────────────────────────────┤
│  rag/llm/              LLM 抽象层                     │
│  ├── base.py           BaseChatClient / BaseEmbedding │
│  │                     / BaseReranker (ABC + 工厂)    │
│  ├── chat.py           OpenAI 兼容 Chat 实现           │
│  ├── embedding.py      远程 Embedding 实现             │
│  └── reranker.py       远程 Reranker 实现              │
├─────────────────────────────────────────────────────┤
│  rag/utils/            存储抽象层                      │
│  ├── doc_store_conn.py DocStoreConnection (ABC + 工厂) │
│  └── es_conn.py        Elasticsearch 实现              │
├─────────────────────────────────────────────────────┤
│  rag/app/              分块引擎 (9 策略 + Router)      │
│  rag/nlp/              NLP (分词/查询扩展/检索)         │
│  rag/graph/            GraphRAG (提取/存储/检索)        │
│  rag/crag/             CRAG (评估/路由/提炼/Web搜索)    │
│  deepdoc/              视觉引擎 (版面分析/OCR/表格识别)  │
└─────────────────────────────────────────────────────┘
```

**三大抽象层**：存储 / LLM / API 全部面向接口编程，替换后端零业务代码改动。

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

## 📡 API 端点

启动后访问 http://localhost:9380/docs 查看 Swagger 文档。

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/health` | GET | 健康检查（ES 状态 + 功能开关） |
| `/api/knowledgebase` | POST | 创建知识库（支持中文名） |
| `/api/knowledgebase` | GET | 列出所有知识库 |
| `/api/knowledgebase/{kb_id}` | DELETE | 删除知识库 |
| `/api/knowledgebase/batch_delete` | POST | 批量删除知识库 |
| `/api/documents/{kb_id}` | GET | 列出知识库文档 |
| `/api/chunks/{kb_id}` | GET | 查看分块内容（分页） |
| `/api/document/{kb_id}/{doc_name}` | DELETE | 删除指定文档 |
| `/api/document/upload` | POST | 上传并解析文档 |
| `/api/retrieval` | POST | 混合检索 + Reranker |
| `/api/graph_retrieval` | POST | GraphRAG + CRAG 增强检索 |

## 📁 项目结构

```
RAGFlow_Lite/
├── api/                        # API 服务层
│   ├── app.py                  # FastAPI 入口壳
│   ├── deps.py                 # 共享依赖注入
│   ├── models.py               # Pydantic 数据模型
│   └── routes/                 # 路由模块
│       ├── kb.py               # 知识库管理
│       ├── doc.py              # 文档管理
│       └── search.py           # 检索服务
├── rag/                        # 核心 RAG 引擎
│   ├── llm/                    # LLM 抽象层
│   │   ├── base.py             # ABC 接口 + 工厂函数
│   │   ├── chat.py             # Chat 客户端
│   │   ├── embedding.py        # Embedding 客户端
│   │   └── reranker.py         # Reranker 客户端
│   ├── nlp/                    # NLP 处理
│   │   ├── search.py           # 混合检索引擎
│   │   ├── query.py            # 查询扩展
│   │   ├── query_enhance.py    # LLM 查询增强
│   │   └── tokenizer.py        # jieba 分词器
│   ├── app/                    # 分块策略引擎
│   │   ├── chunking.py         # Router 工厂
│   │   └── naive|qa|laws|...   # 9 种策略脚本
│   ├── graph/                  # GraphRAG
│   │   ├── extractor.py        # LLM 实体关系提取
│   │   ├── graph_store.py      # 图谱存储 + PageRank
│   │   └── graph_search.py     # 四路并行检索
│   ├── crag/                   # CRAG 纠错增强
│   │   ├── router.py           # 动态路由控制台
│   │   ├── evaluator.py        # 联合裁判器
│   │   ├── refiner.py          # 知识提炼器
│   │   └── web_search.py       # Web 搜索兜底
│   ├── utils/                  # 存储抽象层
│   │   ├── doc_store_conn.py   # ABC 接口 + 工厂
│   │   └── es_conn.py          # Elasticsearch 实现
│   └── parser/                 # 文档解析适配器
├── deepdoc/                    # DeepDoc 视觉引擎
│   ├── parser/                 # PDF/表格解析器
│   └── vision/                 # OCR/版面分析
├── conf/                       # 配置文件
├── scripts/                    # 离线工具
│   └── build_index.py          # 索引构建
├── static/                     # 前端页面
└── docker-compose.yml          # ES Docker
```

## 📜 许可

基于 RAGFlow 开源项目构建，仅供学习研究使用。
