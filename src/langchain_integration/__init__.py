"""LangChain集成模块初始化文件"""
from .qwen_model import QwenLLMWrapper
from .es_vector_store_wrapper import ElasticSearchVectorStore
# LangChainPrompts 现在位于主 prompts.py 中
from ..prompts import LangChainPrompts
from .chains import RAGChain, QuestionRephraseChain, ComparativeAnswerChain

__all__ = ["QwenLLMWrapper", "ElasticSearchVectorStore", "LangChainPrompts", "RAGChain", "QuestionRephraseChain", "ComparativeAnswerChain"]