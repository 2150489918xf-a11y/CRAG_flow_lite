🚨 第一部分：架构融合前的自我审查与排雷 (Self-Critique)
在脑海中将 CRAG（纠错检索增强生成）硬塞进你设计的 4.1~4.6 在线管线时，我发现了 4 个可能导致系统崩溃或性能灾难的逻辑冲突，并以此作为方案优化的基点：

❌ 致命冲突 1：CRAG 传统的“逐块评估”会导致 API 延迟爆炸

原版 CRAG 设想：对 4.2 召回的每一个 Chunk 单独调 LLM 打分。

审查发现：你的系统会召回多个 Chunk 和图谱路径，如果逐个调用外部 DeepSeek API，单次提问的网络延迟将飙升至 10-20 秒，比赛演示时直接翻车。

✅ 修正设计（全局联合评估）：将 CRAG 的拦截点设在 4.4（精排）和 4.5（图谱查完）之后。把最高分的 Top-K Text Chunks 和完整的 Graph CSV 打包成一个联合上下文，只调用 1 次 DeepSeek 充当全局裁判。

❌ 致命冲突 2：图谱数据与文本数据的“语义打架”

审查发现：图谱是高度结构化的（实体、关系），而 Chunk 是长文本。如果评估器只看文本，可能会因为文本没覆盖到细节而误判为 Incorrect；如果混在一起，大模型可能会由于注意力分散而产生幻觉。

✅ 修正设计（双源隔离打分）：在喂给 CRAG 评估器的 Prompt 中，必须使用严格的 XML 标签（<KnowledgeGraph> 和 <TextChunks>）进行物理隔离，让 LLM 基于“图文双重证据”下结论。

❌ 致命冲突 3：引入外网搜索后，原生的 4.6 Prompt 组装会报错

审查发现：你 4.6 步的 Prompt 组装是强依赖本地 ES 字段的（比如需要读取 content_with_weight）。如果 CRAG 触发了外搜，抓回来的是纯文本字典，丢给 4.6 步会直接触发 KeyError 导致程序崩溃。

✅ 修正设计（统一知识总线协议）：外网抓取的数据必须经过一个适配器（Adapter），将其强制伪装（Mock）成你系统中的 Chunk 格式，并赋予虚拟 ID（如 [Web_Tavily_01]），实现底层架构对数据源的无感兼容。

❌ 致命冲突 4：Ambiguous（模糊）状态下的 API 串行灾难

审查发现：按传统流程，若判定为模糊 ➡️ 调 LLM 提炼文本 ➡️ 调 LLM 生成搜索词 ➡️ 去搜网。串行等待太长。

✅ 修正设计（算子折叠）：让评估器在输出 Ambiguous 的同时，强制一并输出一个 search_query（外搜关键词）。然后利用 Python 的 asyncio.gather，将“本地知识提炼”与“外网定向搜索”并发执行，耗时骤降 50%。

🏆 第二部分：RAGFlow Lite + CRAG 详细融合设计图纸
基于上述审查，我们不对你的 1~3（离线阶段）做任何修改。核心改动在于：在 4.5 之后、4.6 之前，插入一个全新的 rag/crag 动态路由控制台中枢。

一、 工程目录调整 (Diff)
在你的 rag/ 目录下新增一个纯 Python 的 CRAG 模块：

Plaintext
RAGFlow_Lite/
├── rag/
│   ├── nlp/ ...
│   ├── graph/ ...
│   ├── crag/                   <--- 【🌟 全新引入模块】
│   │   ├── __init__.py
│   │   ├── evaluator.py        # 核心大脑：联合裁判器
│   │   ├── refiner.py          # 脱水机：提炼 Ambiguous 状态的冗余文本
│   │   ├── web_search.py       # 外网触角：调用 Tavily API / DuckDuckGo
│   │   └── router.py           # 总调度台：基于 IF/ELSE 的状态机流转
├── api/app.py                  <--- 【修改：接入 crag router】
二、 CRAG 核心模块详细设计与代码骨架

1. 核心大脑：联合裁判器 (rag/crag/evaluator.py)
利用 DeepSeek 的强逻辑和 JSON Mode 进行一次性全局审判。

Python
import json
from rag.llm.chat import chat_with_deepseek # 复用你的 LLM 客户端

async def evaluate_context(query: str, text_chunks: list, graph_csv: str) -> dict:
    """对 4.4 和 4.5 的结果进行联合评估"""

    chunks_str = "\n".join([f"[{i}] {c.get('content_with_weight', '')}" for i, c in enumerate(text_chunks)])
    
    prompt = f"""你是一个苛刻的知识核查裁判。请综合评估【本地知识图谱】和【本地文本片段】能否准确、完整地解答【用户问题】。
【用户问题】：{query}

<KnowledgeGraph>
{graph_csv}
</KnowledgeGraph>

<TextChunks>
{chunks_str}
</TextChunks>

请严格按照以下标准评估，并仅输出符合格式的 JSON：

1. "Correct": 提供的知识包含明确、直接的答案依据。
2. "Incorrect": 提供的知识完全无关，或根本无法回答该问题。
3. "Ambiguous": 知识沾边，但缺失核心细节（如缺乏最新数据、具体数值、时间节点等）。

JSON 输出格式：
{{
    "score": "Correct" | "Incorrect" | "Ambiguous",
    "reason": "简短的一句话理由",
    "search_query": "如果得分为 Incorrect 或 Ambiguous，请提取确实信息的关键词，生成一个用于外部搜索引擎的精准短查询（如'微软 2024 投资 OpenAI 金额'）。如果 Correct 则为空字符串。"
}}"""

    # 强制大模型返回 JSON
    response = await chat_with_deepseek(prompt, response_format={"type": "json_object"}, temperature=0.1)
    try:
        return json.loads(response)
    except:
        # 优雅降级兜底
        return {"score": "Ambiguous", "reason": "解析异常兜底", "search_query": query}
2. 外网触角：搜索适配器 (rag/crag/web_search.py)
比赛强烈建议使用 Tavily API（每月免费 1000 次，专为 RAG 优化，直接返回干净摘要）。关键在于格式伪装。

Python
import requests
import os

async def do_web_search(query: str, top_k: int = 3) -> list:
    """调用搜索引擎，并返回伪装成 RAGFlow Lite 标准的 Virtual Chunks"""
    api_key = os.getenv("TAVILY_API_KEY")
    url = "<https://api.tavily.com/search>"
    payload = {"api_key": api_key, "query": query, "search_depth": "basic", "max_results": top_k}

    virtual_chunks = []
    try:
        resp = requests.post(url, json=payload).json()
        for i, res in enumerate(resp.get("results", [])):
            # 🌟 核心：伪装成 ES 查出来的字典格式，适配你的 4.6 组装逻辑
            virtual_chunks.append({
                "chunk_id": f"Web_Tavily_{i+1}",
                "content_with_weight": f"【外网文献：{res['title']}】\n摘要：{res['content']}",
                "source_url": res["url"],
                "knowledge_graph_kwd": "web" # 特殊标记
            })
    except Exception as e:
        print(f"Web Search Failed: {e}")
    return virtual_chunks
3. 知识脱水机：提炼器 (rag/crag/refiner.py)
针对 Ambiguous 状态。本地知识虽然不够，但也不能全盘丢弃，需要把水分抽干，给外网搜回来的数据腾出 Prompt 空间。

Python
from rag.llm.chat import chat_with_deepseek

async def refine_local_knowledge(query: str, text_chunks: list) -> list:
    chunks_str = "\n".join([c.get('content_with_weight', '') for c in text_chunks])

    prompt = f"""请从以下冗长的文档中，提取出所有与问题“{query}”相关的核心客观事实。
去除所有废话、无关段落。用极简的要点(Bullet points)输出。如果毫无用处，输出“无提取事实”。
文档：\n{chunks_str}"""

    refined_text = await chat_with_deepseek(prompt, temperature=0.1)
    
    # 同样伪装成 Virtual Chunk
    return [{
        "chunk_id": "Local_Refined_01",
        "content_with_weight": f"【本地知识提炼事实】\n{refined_text}",
        "knowledge_graph_kwd": "refined"
    }]
4. 总控枢纽：动态路由 (rag/crag/router.py) 🌟🌟🌟
这是 CRAG 的心脏，纯 Python 状态机，无需任何臃肿框架。

Python
import asyncio
from rag.crag.evaluator import evaluate_context
from rag.crag.web_search import do_web_search
from rag.crag.refiner import refine_local_knowledge

async def run_crag_pipeline(query: str, local_chunks: list, graph_csv: str) -> tuple:
    """
    接收 4.4 和 4.5 的原始召回，通过大脑判断，输出最终送给 4.6 的纯净数据。
    返回: (final_chunks_list, final_graph_csv, crag_status_dict)
    """
    # 1. 联合评估大脑介入
    eval_res = await evaluate_context(query, local_chunks, graph_csv)
    score = eval_res.get("score", "Ambiguous")
    search_query = eval_res.get("search_query", query)

    final_chunks = []
    final_graph = graph_csv # 图谱通常包含极高密度的逻辑，默认保留

    # 2. 十字路口：动态路由流转
    if score == "Correct":
        # 🟢 完美命中：不产生冗余操作，原生数据直接放行
        final_chunks = local_chunks
        
    elif score == "Incorrect":
        # 🔴 幻觉熔断：本地全错或无知识。执行焦土政策！
        # 彻底清空本地文本和图谱，防止大模型被垃圾数据误导发生幻觉
        final_graph = "" 
        final_chunks = await do_web_search(search_query)
        
    elif score == "Ambiguous":
        # 🟡 信息残缺：双管齐下 (并发执行，降低延迟)
        # 提炼本地文本 + 用生成的 search_query 去外网精准找补
        refine_task = refine_local_knowledge(query, local_chunks)
        search_task = do_web_search(search_query, top_k=2)
        
        refined_chunks, web_chunks = await asyncio.gather(refine_task, search_task)
        
        final_chunks = refined_chunks + web_chunks

    return final_chunks, final_graph, eval_res
三、 检索流水线串联 (修改 api/app.py 及 4.6 步骤)
在你的 FastAPI 接口中，将管线缝合，并强化 4.6 步的 Prompt 溯源组装：

Python
from fastapi import APIRouter
from rag.crag.router import run_crag_pipeline

# ... 导入你原有的组件

@app.post("/api/graph_retrieval")
async def retrieval_endpoint(request: QueryRequest):
    query = request.query

    # 阶段一：本地极速检索 (原 4.1 ~ 4.5 步)
    expanded_q = query_expander(query)
    hybrid_candidates = es_hybrid_search(expanded_q)
    reranked_chunks = bge_reranker(query, hybrid_candidates) # 4.4
    graph_csv = graph_search(query) # 4.5
    
    # 阶段二：🔥🔥 CRAG 动态路由大脑拦截 🔥🔥
    final_chunks, final_graph, crag_log = await run_crag_pipeline(
        query, reranked_chunks, graph_csv
    )
    
    # 阶段三：强引用 Prompt 组装 (升级你的 4.6 步)
    prompt = f"""你是一个严谨的 AI 专家。请**严格**基于以下提供的【知识图谱】和【多源事实片段】回答问题。
要求：

1. 严禁使用自身常识瞎编。如果提供的资料中没有答案，直接回复“现有资料不足以解答”。
2. 【必须溯源】：在生成每一句话的末尾，必须标注信息来源的 ID。

=== 知识图谱推理 ===
{final_graph if final_graph else "无图谱上下文"}

=== 多源事实片段 ===\n"""

    for c in final_chunks:
        # 读取真实的本地 ID 或伪装的 Web ID
        chunk_id = c.get("chunk_id", "Local_Unknow") 
        content = c.get("content_with_weight", "")
        prompt += f"[ID: {chunk_id}]\n{content}\n\n"
        
    prompt += f"用户问题: {query}"
    
    # 阶段四：LLM 生成
    answer = await deepseek_chat.generate(prompt)
    
    return {
        "answer": answer,
        "crag_score": crag_log["score"],     # 极简前端展示：返回 Correct/Incorrect/Ambiguous 灯
        "crag_reason": crag_log["reason"],   # 展示 AI 思考过程
        "sources": final_chunks              # 返回带 ID 的信源供前端高亮
    }
