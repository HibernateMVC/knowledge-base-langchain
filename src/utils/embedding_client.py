# 向量化工具类
from typing import List, Dict, Any
import dashscope
from src.config.settings import Config
from src.utils.logger import logger

class EmbeddingClient:
    """阿里通义 embedding 客户端"""
    
    def __init__(self):
        # 设置API密钥
        dashscope.api_key = Config.DASHSCOPE_API_KEY
        self.model = Config.EMBEDDING_MODEL_NAME
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入文档列表"""
        try:
            logger.info(f"正在嵌入 {len(texts)} 个文档")
            
            # 使用dashscope TextEmbedding API
            response = dashscope.TextEmbedding.call(
                model=self.model,
                input=texts
            )
            
            if response.status_code == 200:
                # 检查 response.output 是否为字典类型
                if isinstance(response.output, dict):
                    # 如果是字典类型，通过键访问
                    embeddings_data = response.output.get('embeddings', [])
                    embeddings = [item['embedding'] for item in embeddings_data]
                else:
                    # 如果是对象类型，通过属性访问
                    embeddings = [item.embedding for item in response.output.embeddings]
                    
                logger.info(f"文档嵌入完成，每个向量维度: {len(embeddings[0]) if embeddings else 0}")
                return embeddings
            else:
                error_msg = f"Embedding API调用失败: {response.code} - {response.message}"
                logger.error(error_msg)
                raise Exception(error_msg)
                
        except Exception as e:
            logger.error(f"文档嵌入失败: {str(e)}")
            raise
    
    def embed_query(self, query: str) -> List[float]:
        """嵌入查询文本"""
        try:
            logger.info(f"正在嵌入查询: {query[:50]}...")
            
            # 使用dashscope TextEmbedding API
            response = dashscope.TextEmbedding.call(
                model=self.model,
                input=[query]
            )
            
            if response.status_code == 200:
                # 检查 response.output 是否为字典类型
                if isinstance(response.output, dict):
                    # 如果是字典类型，通过键访问
                    embeddings_data = response.output.get('embeddings', [])
                    if embeddings_data:
                        embedding = embeddings_data[0]['embedding']
                    else:
                        raise Exception("API响应中没有embedding数据")
                else:
                    # 如果是对象类型，通过属性访问
                    embedding = response.output.embeddings[0].embedding
                    
                logger.info(f"查询嵌入完成，向量维度: {len(embedding)}")
                return embedding
            else:
                error_msg = f"Embedding API调用失败: {response.code} - {response.message}"
                logger.error(error_msg)
                raise Exception(error_msg)
                
        except Exception as e:
            logger.error(f"查询嵌入失败: {str(e)}")
            raise