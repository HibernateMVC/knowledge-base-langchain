"""阿里 DashScope Embedding 的 LangChain 包装器"""
from typing import List
from langchain_core.embeddings import Embeddings
from src.config.settings import Config
from src.utils.logger import logger
import dashscope


class DashScopeEmbeddings(Embeddings):
    """阿里 DashScope Embedding 模型的 LangChain 包装"""
    
    def __init__(self, model_name: str = None, batch_size: int = 10):
        """
        初始化 DashScope Embeddings
        
        Args:
            model_name: 模型名称，默认使用配置中的模型
            batch_size: 批处理大小，默认 10（DashScope 限制）
        """
        self.model_name = model_name or Config.EMBEDDING_MODEL_NAME
        self.batch_size = batch_size
        dashscope.api_key = Config.DASHSCOPE_API_KEY
        logger.info(f"DashScope Embeddings 初始化: 模型={self.model_name}, 批大小={batch_size}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        批量嵌入文档
        
        Args:
            texts: 文本列表
            
        Returns:
            嵌入向量列表
        """
        if not texts:
            return []
        
        all_embeddings = []
        
        # 分批处理
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            try:
                logger.info(f"正在嵌入文档批次 {i//self.batch_size + 1}/{(len(texts) + self.batch_size - 1)//self.batch_size}")
                
                response = dashscope.TextEmbedding.call(
                    model=self.model_name,
                    input=batch,
                    text_type="document"
                )
                
                if response.status_code == 200:
                    embeddings = [item['embedding'] for item in response.output['embeddings']]
                    all_embeddings.extend(embeddings)
                    logger.info(f"批次嵌入成功，获得 {len(embeddings)} 个向量")
                else:
                    error_msg = f"Embedding API 调用失败: {response.code} - {response.message}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
                    
            except Exception as e:
                logger.error(f"文档嵌入失败: {str(e)}")
                raise
        
        return all_embeddings
    
    def embed_query(self, text: str) -> List[float]:
        """
        嵌入查询文本
        
        Args:
            text: 查询文本
            
        Returns:
            嵌入向量
        """
        try:
            logger.info(f"正在嵌入查询: {text[:50]}...")
            
            response = dashscope.TextEmbedding.call(
                model=self.model_name,
                input=text,
                text_type="query"
            )
            
            if response.status_code == 200:
                embedding = response.output['embeddings'][0]['embedding']
                logger.info(f"查询嵌入完成，向量维度: {len(embedding)}")
                return embedding
            else:
                error_msg = f"Embedding API 调用失败: {response.code} - {response.message}"
                logger.error(error_msg)
                raise Exception(error_msg)
                
        except Exception as e:
            logger.error(f"查询嵌入失败: {str(e)}")
            raise
