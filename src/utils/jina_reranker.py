# Jina重排序器 - 替代BGE重排序器
from typing import List, Dict, Any
from src.utils.logger import logger
import sys
import os


class JinaReranker:
    """
    Jina重排序器 - 使用jinaai/jina-reranker-v1-turbo-en模型或其他替代方案
    用于对检索结果进行精细重排序，提升最终结果的相关性
    """
    
    def __init__(self, model_name: str = "jinaai/jina-reranker-v1-turbo-en"):
        """
        初始化Jina重排序器
        
        Args:
            model_name: Jina重排序模型名称
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._initialized = False
        
    def initialize(self):
        """初始化模型"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            logger.info(f"正在加载Jina重排序模型: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=False)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, local_files_only=False)
            self._initialized = True
            logger.info("Jina重排序模型加载成功")
            
        except ImportError:
            logger.warning("未安装transformers库，Jina重排序功能不可用")
            logger.info("可通过 pip install transformers torch 安装所需依赖")
            self._initialized = False
            
        except OSError:
            # 如果无法下载在线模型，则尝试加载本地模型或其他可用模型
            try:
                logger.info(f"在线模型加载失败，尝试备用模型...")
                # 尝试使用一个较小的模型作为备选
                backup_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
                logger.info(f"正在加载备用重排序模型: {backup_model}")
                from sentence_transformers import CrossEncoder
                self.model = CrossEncoder(backup_model)
                self.model_name = backup_model
                self._initialized = True
                logger.info("备用重排序模型加载成功")
            except Exception:
                logger.warning("备用模型加载也失败，Jina重排序功能将不可用")
                self._initialized = False
                
        except Exception as e:
            logger.error(f"Jina重排序模型加载失败: {str(e)}")
            # 尝试加载备用模型
            try:
                logger.info("尝试加载备用重排序模型...")
                from sentence_transformers import CrossEncoder
                backup_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"
                self.model = CrossEncoder(backup_model)
                self.model_name = backup_model
                self._initialized = True
                logger.info("备用重排序模型加载成功")
            except Exception as backup_error:
                logger.error(f"备用重排序模型加载失败: {str(backup_error)}")
                self._initialized = False
    
    def rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        对文档进行重排序
        
        Args:
            query: 查询文本
            documents: 文档列表
            top_k: 返回前k个最相关的结果
            
        Returns:
            重排序后的结果列表，包含文档索引和相关性得分
        """
        if not self._initialized:
            logger.warning("Jina重排序未初始化，返回原始顺序")
            return [{'index': i, 'score': 1.0/(i+1), 'text': doc} 
                    for i, doc in enumerate(documents[:top_k])]
        
        try:
            # 检查是否使用CrossEncoder模型
            if hasattr(self.model, 'predict'):
                # 对于CrossEncoder模型
                sentence_pairs = [[query, doc] for doc in documents]
                scores = self.model.predict(sentence_pairs)
                
                # 转换得分并排序
                score_list = [(i, float(scores[i])) for i in range(len(documents))]
                score_list.sort(key=lambda x: x[1], reverse=True)
            else:
                # 使用原始transformers方法
                sentence_pairs = [[query, doc] for doc in documents]
                
                # Tokenize 输入
                features = self.tokenizer(
                    sentence_pairs,
                    padding=True,
                    truncation=True,
                    return_tensors='pt',
                    max_length=512
                )
                
                import torch
                # 计算相关性得分
                with torch.no_grad():
                    scores = self.model(**features).logits.view(-1).float()
                    
                # 转换得分并排序
                score_list = [(i, float(scores[i])) for i in range(len(documents))]
                score_list.sort(key=lambda x: x[1], reverse=True)
            
            # 返回top_k结果
            results = []
            for i in range(min(top_k, len(score_list))):
                idx, score = score_list[i]
                results.append({
                    'index': idx,
                    'score': float(score),
                    'text': documents[idx],
                    'original_rank': idx  # 原始排名
                })
            
            logger.debug(f"Jina重排序完成，处理了 {len(documents)} 个文档，返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"Jina重排序失败: {str(e)}")
            # 返回降序排列的原始结果
            return [{'index': i, 'score': 1.0/(i+1), 'text': doc} 
                    for i, doc in enumerate(documents[:top_k])]

    def rerank_search_results(self, query: str, search_results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """
        对搜索结果进行重排序
        
        Args:
            query: 查询文本
            search_results: 搜索结果列表，每个元素包含 'content' 字段
            top_k: 返回前k个最相关的结果
            
        Returns:
            重排序后的搜索结果
        """
        if not search_results:
            return []
            
        # 提取文档内容
        documents = [result['content'] for result in search_results]
        
        # 进行重排序
        reranked_results = self.rerank(query, documents, top_k)
        
        # 重新组装结果，保持原始结果的所有信息
        final_results = []
        for reranked_result in reranked_results:
            original_idx = reranked_result['index']
            original_result = search_results[original_idx].copy()
            # 添加重排序后的得分
            original_result['rerank_score'] = reranked_result['score']
            original_result['rerank_position'] = len(final_results) + 1
            original_result['original_position'] = original_idx + 1
            final_results.append(original_result)
        
        logger.info(f"Jina重排序搜索结果完成，返回 {len(final_results)} 个结果")
        return final_results


# 全局重排序器实例
_jina_reranker = None


def get_jina_reranker(model_name: str = "jinaai/jina-reranker-v1-turbo-en") -> JinaReranker:
    """获取全局Jina重排序器实例"""
    global _jina_reranker
    if _jina_reranker is None:
        _jina_reranker = JinaReranker(model_name)
        _jina_reranker.initialize()
    return _jina_reranker