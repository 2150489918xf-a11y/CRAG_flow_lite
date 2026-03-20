"""
轻量级 common.misc_utils shim
提供 pdf_parser.py 需要的 pip_install_torch() 和 thread_pool_exec()
"""
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)


def pip_install_torch():
    """空实现 — RAGFlow Lite 不自动安装 torch，用户需自行安装（如需 GPU 加速）"""
    pass


def thread_pool_exec(fn, args_list, max_workers=None):
    """
    简单线程池执行
    fn: 可调用对象
    args_list: 参数列表，每个元素是 fn 的一组参数（tuple）
    """
    if not args_list:
        return []

    max_workers = max_workers or min(8, len(args_list))
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for args in args_list:
            if isinstance(args, tuple):
                futures.append(executor.submit(fn, *args))
            else:
                futures.append(executor.submit(fn, args))
        for f in futures:
            try:
                results.append(f.result())
            except Exception as e:
                logger.warning(f"thread_pool_exec task failed: {e}")
                results.append(None)
    return results
