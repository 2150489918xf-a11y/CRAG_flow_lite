"""
统一错误处理层
自定义异常 + 全局异常处理器 + 标准化响应格式
"""
import logging
import time
import traceback
from typing import Any, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════
#  标准响应格式
# ══════════════════════════════════════════

def ok_response(data: Any = None, message: str = "ok", **extra) -> dict:
    """统一成功响应"""
    resp = {"code": 0, "message": message}
    if data is not None:
        resp["data"] = data
    resp.update(extra)
    return resp


def error_response(code: int, message: str, detail: str = "", **extra) -> dict:
    """统一错误响应"""
    resp = {"code": code, "message": message}
    if detail:
        resp["detail"] = detail
    resp.update(extra)
    return resp


# ══════════════════════════════════════════
#  自定义业务异常
# ══════════════════════════════════════════

class AppError(Exception):
    """业务异常基类"""
    def __init__(self, message: str, code: int = 500, detail: str = ""):
        self.message = message
        self.code = code
        self.detail = detail
        super().__init__(message)


class NotFoundError(AppError):
    """资源不存在"""
    def __init__(self, resource: str, identifier: str = ""):
        msg = f"{resource} 不存在"
        if identifier:
            msg = f"{resource} '{identifier}' 不存在"
        super().__init__(msg, code=404)


class ValidationError(AppError):
    """参数校验失败"""
    def __init__(self, message: str):
        super().__init__(message, code=422)


class ExternalServiceError(AppError):
    """外部服务调用失败 (ES / LLM / Reranker)"""
    def __init__(self, service: str, detail: str = ""):
        msg = f"{service} 服务调用失败"
        super().__init__(msg, code=502, detail=detail)


# ══════════════════════════════════════════
#  全局异常处理器
# ══════════════════════════════════════════

def register_error_handlers(app: FastAPI):
    """注册全局异常处理器，保证所有错误都返回统一格式"""

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        logger.warning(f"[{exc.code}] {request.method} {request.url.path}: {exc.message}")
        return JSONResponse(
            status_code=exc.code,
            content=error_response(exc.code, exc.message, exc.detail),
        )

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException):
        logger.warning(f"[{exc.status_code}] {request.method} {request.url.path}: {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content=error_response(exc.status_code, str(exc.detail)),
        )

    @app.exception_handler(Exception)
    async def global_error_handler(request: Request, exc: Exception):
        tb = traceback.format_exc()
        logger.error(f"[500] {request.method} {request.url.path}: {exc}\n{tb}")
        return JSONResponse(
            status_code=500,
            content=error_response(500, "服务器内部错误", str(exc)[:200]),
        )


# ══════════════════════════════════════════
#  请求日志中间件
# ══════════════════════════════════════════

def register_request_logging(app: FastAPI):
    """注册请求日志中间件：记录每个请求的耗时"""

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        elapsed = (time.time() - start) * 1000
        if not request.url.path.startswith("/static"):
            logger.info(
                f"{request.method} {request.url.path} → {response.status_code} ({elapsed:.0f}ms)"
            )
        return response
