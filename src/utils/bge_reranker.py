# 通用重排序器 - 支持多种重排模型
from typing import List, Dict, Any
from src.utils.logger import logger
import os
from dotenv import load_dotenv
from openai import OpenAI
import requests
import src.prompts as prompts
from concurrent.futures import ThreadPoolExecutor


class UniversalBGEReranker:
    """
    通用BGE重排序器，支持多种重排模型和方式
    用于对检索结果进行精细重排序，提升最终结果的相关性
    """
    
    def __init__(self, model_type: str = "jina_api"):
        """
        初始化重排序器
        
        Args:
            model_type: 重排模型类型 ("cross_encoder", "jina_api", "llm")
        """
        self.model_type = model_type.lower()
        self.model = None
        self.llm_client = None
        self.jina_headers = None
        self._initialized = False
        
        # 初始化模型
        self._setup_model()
    
    def _setup_model(self):
        """根据指定的模型类型初始化相应的重排模型"""
        if self.model_type == "cross_encoder":
            self._setup_cross_encoder()
        elif self.model_type == "jina_api":
            self._setup_jina_api()
        elif self.model_type == "llm":
            self._setup_llm()
        else:
            logger.error(f"不支持的模型类型: {self.model_type}")
            # 默认使用交叉编码器
            self.model_type = "cross_encoder"
            self._setup_cross_encoder()
    
    def _setup_cross_encoder(self):
        """初始化CrossEncoder模型"""
        try:
            from sentence_transformers import CrossEncoder
            model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            logger.info(f"正在尝试加载重排序模型: {model_name}")
            
            try:
                # 尝试加载模型
                self.model = CrossEncoder(model_name)
                self._initialized = True
                logger.info("CrossEncoder重排序模型加载成功")
            except Exception as e:
                logger.warning(f"加载CrossEncoder模型时出错: {str(e)}")
                
        except ImportError:
            logger.warning("sentence-transformers库未安装")
            
        except Exception as e:
            logger.error(f"CrossEncoder初始化失败: {str(e)}")
        
        if not self._initialized:
            logger.info("CrossEncoder重排序功能将不可用，系统将继续运行但跳过重排序步骤")
    
    def _setup_jina_api(self):
        """初始化Jina API重排器"""
        try:
            # 初始化Jina重排API地址和请求头
            self.url = 'https://api.jina.ai/v1/rerank'
            self.jina_headers = self._get_jina_headers()
            self._initialized = True
            logger.info("Jina API重排序器初始化成功")
        except Exception as e:
            logger.error(f"Jina API重排序器初始化失败: {str(e)}")
    
    def _get_jina_headers(self):
        """获取Jina API请求头"""
        load_dotenv()
        jina_api_key = os.getenv("JINA_API_KEY")    
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {jina_api_key}'
        }
        return headers
    
    def _setup_llm(self):
        """初始化LLM重排器"""
        # 支持 openai/dashscope，默认 dashscope
        self.provider = os.getenv("LLM_PROVIDER", "dashscope").lower()
        try:
            self.llm_client = self._set_up_llm_client()
            self.system_prompt_rerank_single_block = prompts.RerankingPrompt.system_prompt_rerank_single_block
            self.system_prompt_rerank_multiple_blocks = prompts.RerankingPrompt.system_prompt_rerank_multiple_blocks
            self.schema_for_single_block = prompts.RetrievalRankingSingleBlock
            self.schema_for_multiple_blocks = prompts.RetrievalRankingMultipleBlocks
            self._initialized = True
            logger.info("LLM重排序器初始化成功")
        except Exception as e:
            logger.error(f"LLM重排序器初始化失败: {str(e)}")
    
    def _set_up_llm_client(self):
        """根据 provider 初始化 LLM 客户端"""
        load_dotenv()
        if self.provider == "openai":
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.provider == "dashscope":
            import dashscope
            dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
            return dashscope
        else:
            raise ValueError(f"不支持的 LLM provider: {self.provider}")

    def _cross_encoder_rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """CrossEncoder 重排实现"""
        if not self._initialized:
            logger.debug("CrossEncoder重排序器未初始化，返回原始顺序")
            # 返回按原始顺序的结果，使用简单的分数（按位置分配）
            return [{'index': i, 'score': 1.0/(i+1), 'text': doc, 'original_rank': i, 'rerank_score': 1.0/(i+1)} 
                    for i, doc in enumerate(documents[:top_k])]
        
        try:
            # 使用CrossEncoder模型
            sentence_pairs = [[query, doc] for doc in documents]
            scores = self.model.predict(sentence_pairs)
            
            # 转换得分并排序（按得分降序）
            score_list = [(i, float(scores[i])) for i in range(len(documents))]
            score_list.sort(key=lambda x: x[1], reverse=True)  # 按分数降序排列
        
            # 返回top_k结果
            results = []
            for i in range(min(top_k, len(score_list))):
                idx, score = score_list[i]
                results.append({
                    'index': idx,
                    'score': float(score),
                    'text': documents[idx],
                    'original_rank': idx,  # 原始排名
                    'rerank_score': float(score)  # 重排序得分
                })
            
            logger.debug(f"CrossEncoder重排序完成，处理了 {len(documents)} 个文档，返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"CrossEncoder重排序执行失败: {str(e)}")
            # 返回按原始顺序的结果，但包含rerank_score字段以保持一致性
            return [{'index': i, 'score': 1.0/(i+1), 'text': doc, 'original_rank': i, 'rerank_score': 1.0/(i+1)} 
                    for i, doc in enumerate(documents[:top_k])]

    def _jina_api_rerank(self, query: str, documents: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        """Jina API 重排实现"""
        if not self._initialized or not self.jina_headers:
            logger.debug("Jina API重排序器未初始化，返回原始顺序")
            return [{'index': i, 'score': 1.0/(i+1), 'text': doc, 'original_rank': i, 'rerank_score': 1.0/(i+1)} 
                    for i, doc in enumerate(documents[:top_k])]
        
        try:
            data = {
                "model": "jina-reranker-v2-base-multilingual",
                "query": query,
                "top_n": top_k,
                "documents": documents
            }

            response = requests.post(url=self.url, headers=self.jina_headers, json=data)

            if response.status_code == 200:
                result = response.json()
                results = []
                
                if 'results' in result:
                    for i, item in enumerate(result['results']):
                        doc_idx = item.get('index', i)
                        score = item.get('relevance_score', 0.0)
                        results.append({
                            'index': doc_idx,
                            'score': float(score),
                            'text': documents[doc_idx] if doc_idx < len(documents) else '',
                            'original_rank': doc_idx,
                            'rerank_score': float(score)
                        })
                
                logger.debug(f"Jina API重排序完成，处理了 {len(documents)} 个文档，返回 {len(results)} 个结果")
                return results
            else:
                logger.warning(f"Jina API重排序请求失败: {response.status_code}, {response.text}")
                # 返回原始顺序
                return [{'index': i, 'score': 1.0/(i+1), 'text': doc, 'original_rank': i, 'rerank_score': 1.0/(i+1)} 
                        for i, doc in enumerate(documents[:top_k])]
        except Exception as e:
            logger.error(f"Jina API重排序执行失败: {str(e)}")
            return [{'index': i, 'score': 1.0/(i+1), 'text': doc, 'original_rank': i, 'rerank_score': 1.0/(i+1)} 
                    for i, doc in enumerate(documents[:top_k])]

    def _llm_rerank(self, query: str, documents: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
        """LLM 重排实现"""
        if not self._initialized or not self.llm_client:
            logger.debug("LLM重排序器未初始化，返回原始顺序")
            return [{'index': i, 'score': 1.0/(i+1), 'text': doc, 'original_rank': i, 'rerank_score': 1.0/(i+1)} 
                    for i, doc in enumerate(documents[:top_k])]
        
        try:
            # 使用LLM对多个文本块进行重排
            formatted_blocks = "\n\n---\n\n".join([f'Block {i+1}:\n\n"""{text}"""' for i, text in enumerate(documents)])
            user_prompt = (
                f"Here is the query: \"{query}\"\n\n"
                "Here are the retrieved text blocks:\n"
                f"{formatted_blocks}\n\n"
                f"You should provide exactly {len(documents)} rankings, in order."
            )
            
            if self.provider == "openai":
                completion = self.llm_client.beta.chat.completions.parse(
                    model="gpt-4o-mini-2024-07-18",
                    temperature=0,
                    messages=[
                        {"role": "system", "content": self.system_prompt_rerank_multiple_blocks},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format=self.schema_for_multiple_blocks
                )
                response = completion.choices[0].message.parsed
                response_dict = response.model_dump()
                rankings = response_dict.get('block_rankings', [])
            elif self.provider == "dashscope":
                import dashscope
                messages = [
                    {"role": "system", "content": self.system_prompt_rerank_multiple_blocks},
                    {"role": "user", "content": user_prompt},
                ]
                rsp = self.llm_client.Generation.call(
                    model="qwen-turbo",
                    messages=messages,
                    temperature=0,
                    result_format='message'
                )
                
                # 健壮性检查，防止 rsp 为 None 或非 dict
                if not rsp or not isinstance(rsp, dict):
                    raise RuntimeError(f"DashScope返回None或非dict: {rsp}")
                    
                if 'output' in rsp and 'choices' in rsp['output']:
                    content = rsp['output']['choices'][0]['message']['content']
                    # 这里只返回字符串，后续可按需解析
                    rankings = [{"relevance_score": 0.0, "reasoning": content} for _ in documents]
                else:
                    raise RuntimeError(f"DashScope返回格式异常: {rsp}")
            else:
                raise ValueError(f"不支持的 LLM provider: {self.provider}")
            
            # 构建结果
            results = []
            processed_docs = 0
            for i, doc in enumerate(documents):
                if i < len(rankings):
                    score = rankings[i].get("relevance_score", 0.0)
                else:
                    score = 0.0  # 默认分数
                
                results.append({
                    'index': i,
                    'score': float(score),
                    'text': doc,
                    'original_rank': i,  # 原始排名
                    'rerank_score': float(score)  # 重排序得分
                })
                processed_docs += 1
                
                if processed_docs >= top_k:
                    break
            
            # 按分数排序
            results.sort(key=lambda x: x['rerank_score'], reverse=True)
            
            logger.debug(f"LLM重排序完成，处理了 {len(documents)} 个文档，返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"LLM重排序执行失败: {str(e)}")
            return [{'index': i, 'score': 1.0/(i+1), 'text': doc, 'original_rank': i, 'rerank_score': 1.0/(i+1)} 
                    for i, doc in enumerate(documents[:top_k])]

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
        if not documents:
            return []
        
        if self.model_type == "cross_encoder":
            return self._cross_encoder_rerank(query, documents, top_k)
        elif self.model_type == "jina_api":
            return self._jina_api_rerank(query, documents, top_k)
        elif self.model_type == "llm":
            return self._llm_rerank(query, documents, top_k)
        else:
            # 默认使用CrossEncoder
            return self._cross_encoder_rerank(query, documents, top_k)

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
            # 添加重排序后的得分（覆盖可能已有的值）
            original_result['rerank_score'] = reranked_result['rerank_score']
            original_result['rerank_position'] = len(final_results) + 1
            original_result['original_position'] = original_idx + 1
            final_results.append(original_result)
        
        logger.info(f"重排序搜索结果完成，返回 {len(final_results)} 个结果")
        return final_results


def create_reranker(model_type: str = "jina_api"):
    """
    创建重排序器的工厂函数
    
    Args:
        model_type: 重排模型类型 ("cross_encoder", "jina_api", "llm")
    
    Returns:
        UniversalBGEReranker 实例
    """
    return UniversalBGEReranker(model_type=model_type)


# 兼容性接口，保持原有功能
class HighCompatibilityReranker(UniversalBGEReranker):
    """
    高兼容性重排序器，与旧版接口兼容
    """
    def __init__(self):
        super().__init__(model_type="jina_api")


def get_high_comp_reranker():
    """获取全局高兼容性重排序器实例（兼容原有接口）"""
    return HighCompatibilityReranker()