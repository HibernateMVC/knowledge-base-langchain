"""LangChain兼容的Elasticsearch向量存储包装器"""
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
)
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from src.models.es_vector_store import ElasticSearchClient
from src.utils.logger import logger
from src.config.settings import Config
import numpy as np


class ElasticSearchVectorStore(VectorStore):
    """LangChain兼容的Elasticsearch向量存储包装器"""
    
    # 实测阿里 text-embedding-v4 同步接口单次最大 batch 为 10
    _EMBED_BATCH_SIZE = 10
    
    def __init__(
        self,
        es_client: ElasticSearchClient,
        embedding_function: Optional[Embeddings] = None,
        **kwargs: Any,
    ):
        """
        初始化ElasticSearchVectorStore
        
        Args:
            es_client: ElasticSearch客户端实例
            embedding_function: 嵌入函数
            **kwargs: 其他参数
        """
        self.es_client = es_client
        self.embedding_function = embedding_function
        super().__init__()

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        """
        添加文本到向量存储（批量 embedding 以减少 API 调用次数）
        
        Args:
            texts: 文本列表
            metadatas: 元数据列表
            **kwargs: 其他参数
            
        Returns:
            IDs列表
        """
        if not texts:
            return []

        metadatas = metadatas or []

        # 获取 embedding 函数
        if not self.embedding_function:
            from src.langchain_integration.embeddings import DashScopeEmbeddings
            self.embedding_function = DashScopeEmbeddings()
        
        def _embed_batch(batch: List[str]) -> List[List[float]]:
            return self.embedding_function.embed_documents(batch)

        # 按批量化并获取所有向量
        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), self._EMBED_BATCH_SIZE):
            batch = texts[i : i + self._EMBED_BATCH_SIZE]
            embeddings = _embed_batch(batch)
            all_embeddings.extend(embeddings)

        # 批量写入向量数据库
        docs_to_index = []
        for i, (text, embedding) in enumerate(zip(texts, all_embeddings)):
            metadata = metadatas[i] if i < len(metadatas) else {}
            docs_to_index.append(
                {
                    "content": text,
                    "vector": embedding,
                    "metadata": metadata,
                }
            )

        ids = self.es_client.bulk_add_documents(docs_to_index)

        logger.info(f"成功添加 {len(ids)} 个文档到向量存储")
        return ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        **kwargs: Any,
    ) -> List[Document]:
        """
        相似性搜索
        
        Args:
            query: 查询文本
            k: 返回结果数量
            **kwargs: 其他参数
            
        Returns:
            Document列表
        """
        # 生成查询向量
        if self.embedding_function:
            query_vector = self.embedding_function.embed_query(query)
        else:
            from src.utils.embedding_client import EmbeddingClient
            embedding_client = EmbeddingClient()
            query_vector = embedding_client.embed_query(query)
        
        # 执行搜索
        results = self.es_client.search(
            query_text=query,
            query_vector=query_vector,
            top_k=k
        )
        
        # 转换为LangChain Document格式
        documents = []
        for result in results:
            doc = Document(
                page_content=result.get('content', ''),
                metadata={
                    'id': result.get('id'),
                    'score': result.get('hybrid_score', 0.0),
                    **result.get('metadata', {})
                }
            )
            documents.append(doc)
        
        logger.info(f"相似性搜索完成，返回 {len(documents)} 个结果")
        return documents
    
    def hybrid_search(
        self,
        query_text: str,
        top_k: int = 5,
        use_reranker: bool = True,
        reranker_model: str = "default"
    ) -> List[Dict[str, Any]]:
        """
        混合搜索（结合文本和向量）
        
        Args:
            query_text: 查询文本
            top_k: 返回结果数量
            use_reranker: 是否使用重排序
            reranker_model: 重排序模型类型
            
        Returns:
            搜索结果列表
        """
        # 生成查询向量
        if self.embedding_function:
            query_vector = self.embedding_function.embed_query(query_text)
        else:
            from src.utils.embedding_client import EmbeddingClient
            embedding_client = EmbeddingClient()
            query_vector = embedding_client.embed_query(query_text)
        
        # 执行混合搜索
        results = self.es_client.hybrid_search(
            query_text=query_text,
            query_vector=query_vector,
            top_k=top_k,
            use_reranker=use_reranker
        )
        
        logger.info(f"混合搜索完成，返回 {len(results)} 个结果")
        return results
    
    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> "ElasticSearchVectorStore":
        """
        从文本创建向量存储实例
        
        Args:
            texts: 文本列表
            embedding: 嵌入函数
            metadatas: 元数据列表
            **kwargs: 其他参数
            
        Returns:
            ElasticSearchVectorStore实例
        """
        # 初始化ES客户端
        es_client = ElasticSearchClient()
        
        # 创建实例
        store = cls(es_client=es_client, embedding_function=embedding)
        
        # 添加文本
        store.add_texts(texts, metadatas)
        
        return store
    
    def as_retriever(self, **kwargs):
        """返回检索器实例"""
        from langchain_core.retrievers import BaseRetriever
        from langchain_core.callbacks import CallbackManagerForRetrieverRun
        
        class ElasticsearchRetriever(BaseRetriever):
            vectorstore: ElasticSearchVectorStore
            search_kwargs: dict = {}
            
            def _get_relevant_documents(
                self,
                query: str,
                *,
                run_manager: CallbackManagerForRetrieverRun
            ) -> List[Document]:
                return self.vectorstore.similarity_search(
                    query, 
                    **self.search_kwargs
                )
                
            def add_texts(
                self,
                texts: List[str],
                metadatas: Optional[List[dict]] = None,
            ) -> List[str]:
                return self.vectorstore.add_texts(texts, metadatas)
        
        return ElasticsearchRetriever(vectorstore=self, search_kwargs=kwargs)