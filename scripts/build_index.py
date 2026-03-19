"""
离线索引构建脚本 (含 GraphRAG 图谱提取)
用法:
  python scripts/build_index.py --kb_id my_kb --docs_dir ./data/documents/
  python scripts/build_index.py --kb_id my_kb --docs_dir ./data/ --no-graph  # 跳过图谱
"""
import argparse
import asyncio
import hashlib
import logging
import os
import sys
import time

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.app.chunking import chunk
from rag.llm.embedding import RemoteEmbedding
from rag.utils.es_conn import ESConnection
from rag.nlp.search import index_name
from rag.settings import get_embedding_config, get_rag_config, get_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_index(kb_id, docs_dir, lang="Chinese", enable_graph=True):
    """
    索引构建主流程：
    1. 创建 ES 索引
    2. 扫描文档 → 解析分块 → Embedding → 写入 ES
    3. (可选) 图谱提取 → PageRank → 拍平写入 ES
    """
    logger.info(f"=== 开始构建索引 ===")
    logger.info(f"知识库 ID: {kb_id}")
    logger.info(f"文档目录: {docs_dir}")
    logger.info(f"GraphRAG: {'启用' if enable_graph else '关闭'}")

    # 1. 连接 ES 并创建索引
    es_conn = ESConnection()
    idx = index_name(kb_id)
    es_conn.create_idx(idx)
    logger.info(f"ES 索引就绪: {idx}")

    # 2. 配置 Embedding 模型
    emb_cfg = get_embedding_config()
    emb_mdl = RemoteEmbedding(
        api_key=emb_cfg["api_key"],
        model_name=emb_cfg["model_name"],
        base_url=emb_cfg.get("base_url", ""),
    )
    logger.info(f"Embedding 模型: {emb_cfg['model_name']}")

    # 3. 扫描文档
    supported_exts = {".pdf", ".docx", ".doc", ".xlsx", ".xls", ".md", ".markdown",
                      ".pptx", ".ppt", ".txt", ".html", ".htm", ".json", ".csv"}

    files = []
    for root, dirs, filenames in os.walk(docs_dir):
        for fn in filenames:
            ext = os.path.splitext(fn)[-1].lower()
            if ext in supported_exts:
                files.append(os.path.join(root, fn))

    logger.info(f"找到 {len(files)} 个文档")

    if not files:
        logger.warning("没有找到可处理的文档")
        return

    # 4. 逐个文件: 分块 → Embedding → 写入 ES
    total_chunks = 0
    total_tokens = 0
    all_chunks_for_graph = []  # 用于后续图谱提取

    for file_idx, fpath in enumerate(files):
        logger.info(f"[{file_idx + 1}/{len(files)}] 处理: {os.path.basename(fpath)}")

        doc_id = hashlib.md5(fpath.encode()).hexdigest()[:16]
        chunks = chunk(fpath, lang=lang)

        if not chunks:
            logger.warning(f"  跳过 (无内容): {fpath}")
            continue

        for ck in chunks:
            ck["doc_id"] = doc_id
            ck["kb_id"] = kb_id
            ck["knowledge_graph_kwd"] = "chunk"  # 标记为普通文本 chunk

        # Embedding
        texts = [ck.get("content_with_weight", "") or " " for ck in chunks]
        logger.info(f"  分块: {len(chunks)}，开始 Embedding...")

        batch_size = 16
        for i in range(0, len(chunks), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_chunks = chunks[i:i + batch_size]

            try:
                embeddings, tokens = emb_mdl.encode(batch_texts)
                total_tokens += tokens
                for ck, emb in zip(batch_chunks, embeddings):
                    dim = len(emb)
                    ck[f"q_{dim}_vec"] = emb.tolist()
            except Exception as e:
                logger.error(f"  Embedding 失败: {e}")
                continue
            time.sleep(0.1)

        # 写入 ES
        errors = es_conn.insert(chunks, idx)
        if errors:
            logger.warning(f"  ES 写入警告: {errors[:3]}")
        else:
            logger.info(f"  写入 {len(chunks)} 个 chunks 到 ES")

        total_chunks += len(chunks)
        all_chunks_for_graph.extend(chunks)

    # 刷新索引
    try:
        es_conn.es.indices.refresh(index=idx)
    except Exception:
        pass

    logger.info(f"--- 文本索引完成: {len(files)} 文档, {total_chunks} chunks, {total_tokens} tokens ---")

    # 5. GraphRAG 图谱提取
    if enable_graph and all_chunks_for_graph:
        _build_graph(kb_id, idx, all_chunks_for_graph, es_conn, emb_mdl)

    logger.info(f"=== 索引构建完成 ===")


def _build_graph(kb_id, idx, chunks, es_conn, emb_mdl):
    """GraphRAG 图谱构建子流程"""
    logger.info(f"\n=== 开始 GraphRAG 图谱构建 ===")

    from rag.llm.chat import ChatClient
    from rag.graph.extractor import GraphExtractor
    from rag.graph.graph_store import GraphStore

    # 初始化
    chat = ChatClient()
    extractor = GraphExtractor(chat)
    graph_store = GraphStore(es_conn=es_conn, emb_mdl=emb_mdl)

    # Step 1: LLM 实体/关系提取
    logger.info(f"从 {len(chunks)} 个 chunks 中提取实体和关系...")
    extraction = extractor.extract_batch(chunks)
    logger.info(f"提取完成: {len(extraction.entities)} 实体, {len(extraction.relations)} 关系")

    if not extraction.entities and not extraction.relations:
        logger.warning("未提取到任何实体或关系，跳过图谱构建")
        return

    # Step 2: 构建 NetworkX 图 + PageRank
    graph_store.build_graph(extraction)
    pagerank = graph_store.compute_pagerank()

    # Step 3: 存入 ES（拍平为 entity/relation 文档）
    logger.info("将图谱数据写入 ES...")
    count = asyncio.run(graph_store.store_to_es(idx, kb_id, extraction, pagerank))
    logger.info(f"写入 {count} 条图谱文档到 ES")

    # Step 4: 保存 NetworkX 图到文件（用于在线 N 跳遍历）
    graph_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "data", "graphs")
    os.makedirs(graph_dir, exist_ok=True)
    graph_path = os.path.join(graph_dir, f"{kb_id}_graph.json")
    graph_store.save_graph(graph_path)

    # 刷新索引
    try:
        es_conn.es.indices.refresh(index=idx)
    except Exception:
        pass

    logger.info(f"=== GraphRAG 图谱构建完成 ===")
    logger.info(f"  节点: {graph_store.graph.number_of_nodes()}")
    logger.info(f"  边: {graph_store.graph.number_of_edges()}")
    logger.info(f"  图文件: {graph_path}")


def main():
    parser = argparse.ArgumentParser(description="RAGFlow Lite 离线索引构建")
    parser.add_argument("--kb_id", required=True, help="知识库 ID")
    parser.add_argument("--docs_dir", required=True, help="文档目录路径")
    parser.add_argument("--lang", default="Chinese", help="语言 (Chinese/English)")
    parser.add_argument("--no-graph", action="store_true", help="跳过 GraphRAG 图谱提取")
    args = parser.parse_args()

    if not os.path.isdir(args.docs_dir):
        logger.error(f"文档目录不存在: {args.docs_dir}")
        sys.exit(1)

    # 检查配置中是否启用 graph
    cfg = get_config()
    graph_enabled = cfg.get("graph", {}).get("enabled", True)
    enable_graph = graph_enabled and not args.no_graph

    build_index(args.kb_id, args.docs_dir, args.lang, enable_graph)


if __name__ == "__main__":
    main()
