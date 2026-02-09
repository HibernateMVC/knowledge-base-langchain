# 日志记录和调试工具
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import json
import traceback
from functools import wraps

def setup_logging():
    """设置应用程序日志"""
    # 创建logs目录
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # 配置基本日志设置
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 主日志文件
    main_log = str(log_dir / f"knowledge_base_{datetime.now().strftime('%Y%m%d')}.log")
    
    # 错误日志文件
    error_log = str(log_dir / f"error_{datetime.now().strftime('%Y%m%d')}.log")
    
    # 配置日志记录器
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler(main_log, encoding='utf-8'),
            logging.StreamHandler(),  # 控制台输出
        ]
    )
    
    # 单独配置错误日志
    error_handler = logging.FileHandler(error_log, encoding='utf-8')
    error_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter(log_format)
    error_handler.setFormatter(error_formatter)
    
    # 添加错误处理器到根日志记录器
    root_logger = logging.getLogger()
    root_logger.addHandler(error_handler)
    
    return root_logger


def log_api_call(func):
    """装饰器：记录API调用信息"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        start_time = datetime.now()
        
        try:
            logger.info(f"开始执行 {func.__name__}")
            logger.debug(f"参数: args={args}, kwargs={kwargs}")
            
            result = func(*args, **kwargs)
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"{func.__name__} 执行完成，耗时: {duration:.2f} 秒")
            logger.debug(f"返回值: {result}")
            
            return result
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            logger.error(f"{func.__name__} 执行失败，耗时: {duration:.2f} 秒")
            logger.error(f"错误详情: {str(e)}")
            logger.error(f"堆栈跟踪: {traceback.format_exc()}")
            raise
    
    return wrapper


def log_operation(operation_name: str):
    """装饰器工厂：记录特定操作的日志"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            start_time = datetime.now()
            
            logger.info(f"[{operation_name}] 开始执行 {func.__name__}")
            
            try:
                result = func(*args, **kwargs)
                
                duration = (datetime.now() - start_time).total_seconds()
                logger.info(f"[{operation_name}] {func.__name__} 执行成功，耗时: {duration:.2f}s")
                
                return result
            except Exception as e:
                duration = (datetime.now() - start_time).total_seconds()
                logger.error(f"[{operation_name}] {func.__name__} 执行失败，耗时: {duration:.2f}s")
                logger.error(f"[{operation_name}] 错误详情: {str(e)}")
                raise
        
        return wrapper
    return decorator


def log_embeddings_info(texts: list, embeddings: list):
    """记录嵌入信息"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"嵌入计算完成，文档数量: {len(texts)}")
    if embeddings:
        logger.info(f"向量维度: {len(embeddings[0]) if embeddings else 0}")
        logger.debug(f"第一个向量的前5个值: {embeddings[0][:5] if embeddings else []}")


def log_search_results(query: str, results: list, method: str = "hybrid"):
    """记录搜索结果信息"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"{method} 搜索 '{query[:50]}...' 返回 {len(results)} 个结果")
    
    if results:
        scores = [r.get('hybrid_score', r.get('score', 0)) for r in results]
        logger.debug(f"最高分数: {max(scores):.4f}, 最低分数: {min(scores):.4f}")


def log_model_interaction(prompt: str, response: str, model_name: str = "qwen"):
    """记录与大模型的交互"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"向 {model_name} 模型发送请求")
    logger.debug(f"输入长度: {len(prompt)} 字符")
    logger.debug(f"输出长度: {len(response)} 字符")
    logger.debug(f"输入预览: {prompt[:100]}..." if len(prompt) > 100 else prompt)


def log_performance_stats(stats: Dict[str, Any]):
    """记录性能统计信息"""
    logger = logging.getLogger(__name__)
    
    logger.info("=== 性能统计 ===")
    for key, value in stats.items():
        logger.info(f"{key}: {value}")
    logger.info("===============")