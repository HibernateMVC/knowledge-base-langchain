# 向量数据库客户端 - ElasticSearch
from elasticsearch import Elasticsearch
from typing import List, Dict, Any, Optional
import sys
import os
# 添加项目根目录到路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import Config
from utils.logger import logger
from utils.es_compatibility import create_es_client, adjust_mapping_for_es9
import numpy as np

# 导入高兼容性重排序器
try:
    from utils.bge_reranker import get_high_comp_reranker
except ImportError as e:
    logger.warning(f"无法导入高兼容性重排序器: {e}")
    get_high_comp_reranker = None

class ElasticSearchClient:
    """ElasticSearch 向量数据库客户端"""
    
    def __init__(self):
        # 使用兼容ES 9.x的方式创建客户端
        self.es = create_es_client()
        self.index_name = Config.ES_INDEX_NAME
        
    def create_index(self, dimension: int = 1536):
        """创建向量索引"""
        try:
            if self.es.indices.exists(index=self.index_name):
                logger.info(f"索引 {self.index_name} 已存在")
                return True
                
            mapping = {
                "mappings": {
                    "properties": {
                        "content": {
                            "type": "text",
                            "analyzer": "standard"
                        },
                        "vector": {
                            "type": "dense_vector",
                            "dims": dimension,
                            "index": True,
                            "similarity": "cosine"
                        },
                        "metadata": {
                            "type": "object",
                            "properties": {
                                "source": {
                                    "type": "keyword"
                                },
                                "filename": {
                                    "type": "keyword"
                                },
                                "page": {
                                    "type": "integer"
                                },
                                "title": {
                                    "type": "text"
                                },
                                "author": {
                                    "type": "keyword"
                                },
                                "subject": {
                                    "type": "text"
                                },
                                "creator": {
                                    "type": "keyword"
                                },
                                "producer": {
                                    "type": "keyword"
                                },
                                "creationDate": {
                                    "type": "date"
                                },
                                "modificationDate": {
                                    "type": "date"
                                },
                                "keywords": {
                                    "type": "text"
                                }
                            }
                        }
                    }
                }
            }
            
            # 为ES 9.x调整映射，避免添加不兼容的参数
            mapping = adjust_mapping_for_es9(mapping)
            
            result = self.es.indices.create(
                index=self.index_name,
                body=mapping
            )
            logger.info(f"成功创建索引 {self.index_name}")
            return result.get('acknowledged', False)
            
        except Exception as e:
            logger.error(f"创建索引失败: {str(e)}")
            raise
            
            result = self.es.indices.create(
                index=self.index_name,
                body=mapping
            )
            logger.info(f"成功创建索引 {self.index_name}")
            return result.get('acknowledged', False)
            
        except Exception as e:
            logger.error(f"创建索引失败: {str(e)}")
            raise
    
    def add_document(self, content: str, vector: List[float], metadata: Dict[str, Any] = None):
        """添加文档到索引"""
        try:
            doc = {
                "content": content,
                "vector": vector,
                "metadata": metadata or {}
            }
            
            response = self.es.index(
                index=self.index_name,
                body=doc
            )
            
            logger.debug(f"文档添加成功: {response.get('_id')}")
            return response
            
        except Exception as e:
            logger.error(f"添加文档失败: {str(e)}")
            raise
    
    def search_by_vector(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """基于向量相似度搜索"""
        try:
            query = {
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                            "params": {
                                "query_vector": query_vector
                            }
                        }
                    }
                },
                "size": top_k
            }
            
            response = self.es.search(
                index=self.index_name,
                body=query
            )
            
            results = []
            for hit in response['hits']['hits']:
                results.append({
                    'id': hit['_id'],
                    'score': hit['_score'],
                    'content': hit['_source']['content'],
                    'metadata': hit['_source']['metadata']
                })
                
            logger.debug(f"向量搜索返回了 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"向量搜索失败: {str(e)}")
            raise
    
    def keyword_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """基于关键词搜索"""
        try:
            query_body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["content^2", "metadata.source", "metadata.filename", "metadata.title", "metadata.author", "metadata.subject", "metadata.creator", "metadata.producer", "metadata.keywords"],  # 只能查询文本和关键字类型字段
                        "type": "best_fields",
                        "fuzziness": "AUTO"
                    }
                },
                "size": top_k
            }
            
            response = self.es.search(
                index=self.index_name,
                body=query_body
            )
            
            results = []
            for hit in response['hits']['hits']:
                results.append({
                    'id': hit['_id'],
                    'score': hit['_score'],
                    'content': hit['_source']['content'],
                    'metadata': hit['_source']['metadata']
                })
                
            logger.debug(f"关键词搜索返回了 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"关键词搜索失败: {str(e)}")
            raise
    
    def hybrid_search(self, query_text: str, query_vector: List[float], 
                     top_k: int = 5, vector_weight: float = 0.7, 
                     use_reranker: bool = True) -> List[Dict[str, Any]]:
        """混合搜索（向量+关键词）"""
        try:
            # 分别执行向量搜索和关键词搜索
            vector_results = self.search_by_vector(query_vector, top_k=top_k*2)
            keyword_results = self.keyword_search(query_text, top_k=top_k*2)
            
            # 将搜索结果合并去重
            all_results = {}
            
            # 添加向量搜索结果，带权重
            for item in vector_results:
                doc_id = item['id']
                all_results[doc_id] = {
                    'id': doc_id,
                    'content': item['content'],
                    'metadata': item['metadata'],
                    'vector_score': item['score'],
                    'keyword_score': 0.0
                }
            
            # 添加/更新关键词搜索结果分数
            for item in keyword_results:
                doc_id = item['id']
                if doc_id not in all_results:
                    all_results[doc_id] = {
                        'id': doc_id,
                        'content': item['content'],
                        'metadata': item['metadata'],
                        'vector_score': 0.0,
                        'keyword_score': item['score']
                    }
                else:
                    all_results[doc_id]['keyword_score'] = item['score']
            
            # 计算综合得分
            pre_rerank_results = []
            for doc_id, scores in all_results.items():
                hybrid_score = (
                    scores['vector_score'] * vector_weight + 
                    scores['keyword_score'] * (1 - vector_weight)
                )
                
                pre_rerank_results.append({
                    'id': scores['id'],
                    'content': scores['content'],
                    'metadata': scores['metadata'],
                    'hybrid_score': hybrid_score,
                    'vector_score': scores['vector_score'],
                    'keyword_score': scores['keyword_score']
                })
            
            # 按综合得分排序
            pre_rerank_results.sort(key=lambda x: x['hybrid_score'], reverse=True)
            
            # 应用高兼容性重排序（如果启用且可用）
            if use_reranker and get_high_comp_reranker is not None:
                try:
                    reranker = get_high_comp_reranker()
                    if reranker and reranker._initialized:
                        logger.info(f"应用高兼容性重排序，对 {len(pre_rerank_results)} 个结果进行重排序")
                        reranked_results = reranker.rerank_search_results(
                            query_text, 
                            pre_rerank_results, 
                            top_k=min(top_k*2, len(pre_rerank_results))
                        )
                        
                        # 对重排序后的结果再次按重排序得分排序
                        reranked_results.sort(key=lambda x: x['rerank_score'], reverse=True)
                        
                        logger.info(f"重排序完成，返回 {len(reranked_results[:top_k])} 个最终结果")
                        return reranked_results[:top_k]
                    else:
                        logger.debug("高兼容性重排序器未初始化，使用原始混合排序结果")
                except Exception as e:
                    logger.error(f"重排序过程中发生错误: {str(e)}，使用原始混合排序结果")
            
            # 如果不使用重排序或重排序失败，返回原始混合排序结果
            final_results = pre_rerank_results[:top_k]
            logger.info(f"混合搜索返回了 {len(final_results)} 个结果")
            return final_results
            
        except Exception as e:
            logger.error(f"混合搜索失败: {str(e)}")
            raise
    
    def delete_index(self):
        """删除索引"""
        try:
            if self.es.indices.exists(index=self.index_name):
                result = self.es.indices.delete(index=self.index_name)
                logger.info(f"成功删除索引 {self.index_name}")
                return result.get('acknowledged', False)
            else:
                logger.warning(f"索引 {self.index_name} 不存在")
                return True
        except Exception as e:
            logger.error(f"删除索引失败: {str(e)}")
            raise