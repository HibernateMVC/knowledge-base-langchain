"""统一的装饰器和工具函数"""
from functools import wraps
from src.utils.logger import logger
from typing import Callable, Any


def log_and_handle_errors(func: Callable) -> Callable:
    """统一的日志和错误处理装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        try:
            logger.info(f"执行 {func.__name__}...")
            result = func(*args, **kwargs)
            logger.info(f"{func.__name__} 执行成功")
            return result
        except Exception as e:
            logger.error(f"{func.__name__} 执行失败: {str(e)}")
            raise
    return wrapper


def log_performance(func: Callable) -> Callable:
    """性能日志装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        import time
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(f"{func.__name__} 耗时: {elapsed:.2f}秒")
            return result
        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"{func.__name__} 失败，耗时: {elapsed:.2f}秒，错误: {str(e)}")
            raise
    return wrapper
