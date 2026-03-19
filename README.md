# RAGFlow Lite

轻量化 RAG 系统，基于 [RAGFlow](https://github.com/infiniflow/ragflow) 核心检索架构构建。

## 架构特点

- ✅ **混合检索**：ES fulltext query_string + KNN 向量检索，融合权重 0.05:0.95
- ✅ **智能分块**：naive merge + token 控制，支持 8 种文档格式
- ✅ **查询扩展**：分词加权 + 同义词 + IDF 优化的 ES query_string
- ✅ **重排序**：hybrid_similarity（cosine * vtweight + token_similarity * tkweight）
- ❌ 无 Docker 25GB 全家桶，只需要一个 ES 实例

## 快速开始

### 1. 启动 Elasticsearch

```bash
docker-compose up -d
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置

编辑 `conf/service_conf.yaml`，配置 Embedding API：

```yaml
embedding:
  api_key: "your-api-key-here"
  model_name: "BAAI/bge-m3"
  base_url: "https://api.siliconflow.cn/v1"
```

### 4. 构建索引

```bash
python scripts/build_index.py --kb_id my_kb --docs_dir ./data/documents/
```

### 5. 启动 API 服务

```bash
python -m api.app
# 或
uvicorn api.app:app --host 0.0.0.0 --port 9380 --reload
```

### 6. 检索

```bash
curl -X POST http://localhost:9380/api/retrieval \
  -H "Content-Type: application/json" \
  -d '{"question": "你好", "kb_ids": ["my_kb"], "top_k": 5}'
```

## API 文档

启动服务后访问 http://localhost:9380/docs 查看 Swagger 文档。

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/health` | GET | 健康检查 |
| `/api/knowledgebase` | POST | 创建知识库 |
| `/api/knowledgebase` | GET | 列出知识库 |
| `/api/knowledgebase/{id}` | DELETE | 删除知识库 |
| `/api/document/upload` | POST | 上传并解析文档 |
| `/api/retrieval` | POST | 混合检索 |

## 项目结构

```
RAGFlow_Lite/
├── conf/                    # 配置
│   ├── mapping.json         # ES 索引映射
│   └── service_conf.yaml    # 服务配置
├── rag/                     # 核心 RAG 模块
│   ├── nlp/                 # NLP 处理
│   │   ├── tokenizer.py     # jieba 分词器
│   │   ├── term_weight.py   # 词权重
│   │   ├── synonym.py       # 同义词
│   │   ├── query.py         # 查询扩展
│   │   └── search.py        # 混合检索
│   ├── parser/              # 文档解析器
│   ├── app/chunking.py      # 分块引擎
│   ├── llm/embedding.py     # Embedding 客户端
│   └── utils/es_conn.py     # ES 连接器
├── api/app.py               # FastAPI 服务
├── scripts/build_index.py   # 离线索引构建
├── res/                     # NLP 资源文件
├── docker-compose.yml       # ES Docker
└── requirements.txt
```
