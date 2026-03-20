"""
轻量级性能监控模块
- 记录各管道阶段的耗时 (ms)
- 滑动窗口统计 (最近 100 次)
- 线程安全
"""
import time
import threading
from collections import defaultdict, deque
from contextlib import contextmanager
from typing import Optional


class PerfCollector:
    """管道耗时收集器，线程安全"""

    def __init__(self, window_size: int = 100):
        self._window_size = window_size
        self._data: dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._lock = threading.Lock()
        self._counters: dict[str, int] = defaultdict(int)

    def record(self, stage: str, elapsed_ms: float):
        """记录一次耗时"""
        with self._lock:
            self._data[stage].append(elapsed_ms)
            self._counters[stage] += 1

    @contextmanager
    def timer(self, stage: str):
        """上下文管理器：自动计时并记录"""
        start = time.perf_counter()
        yield
        elapsed = (time.perf_counter() - start) * 1000
        self.record(stage, elapsed)

    def get_stats(self) -> dict:
        """获取所有阶段的统计信息"""
        with self._lock:
            result = {}
            for stage, times in self._data.items():
                if not times:
                    continue
                sorted_t = sorted(times)
                n = len(sorted_t)
                result[stage] = {
                    "count": self._counters[stage],
                    "recent": n,
                    "avg_ms": round(sum(sorted_t) / n, 1),
                    "min_ms": round(sorted_t[0], 1),
                    "max_ms": round(sorted_t[-1], 1),
                    "p50_ms": round(sorted_t[n // 2], 1),
                    "p95_ms": round(sorted_t[min(int(n * 0.95), n - 1)], 1),
                    "last_ms": round(times[-1], 1),
                }
            return result

    def get_last_request(self) -> dict:
        """获取最近一次请求各阶段耗时"""
        with self._lock:
            return {stage: round(times[-1], 1) for stage, times in self._data.items() if times}

    def reset(self):
        with self._lock:
            self._data.clear()
            self._counters.clear()


# 全局单例
perf = PerfCollector()
