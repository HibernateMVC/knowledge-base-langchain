# 项目配置文件
import os
from typing import Optional

class Config:
    # 通义千问API配置
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
    QWEN_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "qwen-max")
    
    # 阿里embedding模型配置
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v4")
    
    # Elasticsearch配置
    ES_HOST = os.getenv("ES_HOST", "localhost")
    ES_PORT = int(os.getenv("ES_PORT", 9200))
    ES_SCHEME = os.getenv("ES_SCHEME", "https")  # 默认使用https
    ES_USERNAME = os.getenv("ES_USERNAME", "elastic")
    ES_PASSWORD = os.getenv("ES_PASSWORD", "your_elasticsearch_password")
    ES_INDEX_NAME = os.getenv("ES_INDEX_NAME", "knowledge_base_index")
    
    # 服务配置
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8080))  # 修改为8080端口避免冲突
    
    # 数据目录
    UPLOAD_DIR = os.path.join("data", "uploads")
    KNOWLEDGE_DIR = os.path.join("data", "knowledge")
    
    # 日志配置
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.path.join("logs", "app.log")
    
    @classmethod
    def validate(cls):
        """验证配置是否完整"""
        if not cls.DASHSCOPE_API_KEY:
            raise ValueError("DASHSCOPE_API_KEY 环境变量未设置")