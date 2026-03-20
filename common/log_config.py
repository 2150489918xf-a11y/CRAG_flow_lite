"""
统一日志配置模块
- 统一格式: 时间 | 级别 | 模块 | 请求ID | 消息
- 支持从 service_conf.yaml 读取日志级别
- 请求 Trace ID 透传
- 第三方库（ES/httpx/urllib3）自动降噪
"""
import logging
import sys
import uuid
import contextvars

# ══════════════════════════════════════════
#  请求 Trace ID（协程安全）
# ══════════════════════════════════════════

_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")


def get_trace_id() -> str:
    return _trace_id.get()


def set_trace_id(trace_id: str = ""):
    """设置当前协程的 trace_id，为空则自动生成 8 位短 ID"""
    _trace_id.set(trace_id or uuid.uuid4().hex[:8])


def clear_trace_id():
    _trace_id.set("-")


# ══════════════════════════════════════════
#  自定义 Formatter
# ══════════════════════════════════════════

class TraceFormatter(logging.Formatter):
    """带 Trace ID 的统一日志格式"""

    FORMAT = "%(asctime)s │ %(levelname)-5s │ %(name)-28s │ %(trace_id)s │ %(message)s"
    DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self):
        super().__init__(fmt=self.FORMAT, datefmt=self.DATE_FMT)

    def format(self, record: logging.LogRecord) -> str:
        record.trace_id = _trace_id.get()
        # 截断过长的模块名
        if len(record.name) > 28:
            record.name = "..." + record.name[-25:]
        return super().format(record)


# ══════════════════════════════════════════
#  初始化
# ══════════════════════════════════════════

_initialized = False


def setup_logging(level: str = "INFO"):
    """
    初始化全局日志配置。

    Args:
        level: 日志级别 (DEBUG/INFO/WARNING/ERROR)
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    log_level = getattr(logging, level.upper(), logging.INFO)

    # Root handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(TraceFormatter())

    root = logging.getLogger()
    root.setLevel(log_level)
    # 清除已有 handler，防止重复
    root.handlers.clear()
    root.addHandler(handler)

    # ── 第三方库降噪 ──
    noisy_loggers = [
        "elasticsearch",
        "urllib3",
        "httpx",
        "httpcore",
        "uvicorn.access",
        "filelock",
        "sentence_transformers",
        "huggingface_hub",
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)

    # uvicorn 自身日志保持 INFO
    logging.getLogger("uvicorn").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)

    logging.getLogger(__name__).info(f"日志系统初始化完成 (level={level})")
