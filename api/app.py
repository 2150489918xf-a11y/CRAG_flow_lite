"""
RAGFlow Lite FastAPI 应用入口
职责：仅负责创建 FastAPI 实例、注册中间件、挂载路由
"""
import os
# HuggingFace 镜像加速（DeepDoc 模型下载）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import logging
import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

# 确保项目根目录在 Python 路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="RAGFlow Lite", version="0.4.0",
              description="轻量化 RAG 检索服务 (混合检索 + GraphRAG + CRAG)")

# 挂载静态前端文件
_static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir, html=True), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册统一错误处理 + 请求日志
from api.errors import register_error_handlers, register_request_logging
register_error_handlers(app)
register_request_logging(app)


# ==================== 注册路由 ====================

from api.routes.kb import router as kb_router
from api.routes.doc import router as doc_router
from api.routes.search import router as search_router

app.include_router(kb_router)
app.include_router(doc_router)
app.include_router(search_router)


@app.get("/")
async def root():
    """根路径重定向到前端页面"""
    return RedirectResponse(url="/static/index.html")


# ==================== 启动 ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=9380, reload=True)
