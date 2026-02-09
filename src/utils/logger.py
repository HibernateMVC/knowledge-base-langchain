# 日志配置
import os
from loguru import logger
from datetime import datetime

def setup_logger():
    """设置日志记录器"""
    # 清除默认的日志处理器
    logger.remove()
    
    # 添加控制台处理器
    logger.add(
        sink=lambda msg: print(msg, end=''),
        level="INFO",
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # 添加文件处理器
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"kb_system_{datetime.now().strftime('%Y%m%d')}.log")
    
    logger.add(
        sink=log_file,
        level="DEBUG",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}"
    )
    
    return logger

# 创建全局日志记录器
logger = setup_logger()

__all__ = ["logger"]