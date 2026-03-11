# 项目配置文件
import os
from typing import Optional

class Config:
    # 通义千问API配置
    DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "")
    QWEN_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "qwen-turbo")
    
    # 阿里embedding模型配置
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "text-embedding-v4")
    EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "10"))
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
    
    # Elasticsearch配置
    ES_HOST = os.getenv("ES_HOST", "localhost")
    ES_PORT = int(os.getenv("ES_PORT", 9200))
    ES_SCHEME = os.getenv("ES_SCHEME", "https")  # 默认使用https
    ES_USERNAME = os.getenv("ES_USERNAME", "elastic")
    ES_PASSWORD = os.getenv("ES_PASSWORD", "your_elasticsearch_password")
    ES_INDEX_NAME = os.getenv("ES_INDEX_NAME", "knowledge_base_index")
    
    # 服务配置
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8080))
    
    # 数据目录
    UPLOAD_DIR = os.path.join("data", "uploads")
    KNOWLEDGE_DIR = os.path.join("data", "knowledge")
    
    # 日志配置
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.path.join("logs", "app.log")

    # RAG 配置
    RAG_MIN_RELEVANCE_SCORE = float(os.getenv("RAG_MIN_RELEVANCE_SCORE", "0.35"))
    
    # 检索配置
    VECTOR_SEARCH_WEIGHT = float(os.getenv("VECTOR_SEARCH_WEIGHT", "0.6"))
    KEYWORD_SEARCH_WEIGHT = float(os.getenv("KEYWORD_SEARCH_WEIGHT", "0.4"))
    RERANKER_TOP_K = int(os.getenv("RERANKER_TOP_K", "10"))
    
    # LLM 配置
    LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2000"))
    
    # 对话配置
    CONVERSATION_MAX_HISTORY = int(os.getenv("CONVERSATION_MAX_HISTORY", "10"))
    CONVERSATION_CONTEXT_LIMIT = int(os.getenv("CONVERSATION_CONTEXT_LIMIT", "4000"))
    
    # 重排序配置
    RERANKER_MODEL = os.getenv("RERANKER_MODEL", "bge")
    
    @classmethod
    def validate(cls):
        """验证配置是否完整"""
        if not cls.DASHSCOPE_API_KEY:
            raise ValueError("DASHSCOPE_API_KEY 环境变量未设置")