# 知识库主类
from typing import List, Dict, Any
import sys
import os
# 添加项目根目录到路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.es_vector_store import ElasticSearchClient
from utils.embedding_client import EmbeddingClient
from utils.document_loader import DocumentLoader
from utils.qwen_client import QwenLLMClient
from utils.logger import logger
from src.prompts import (
    AnswerWithRAGContextNamePrompt,
    AnswerWithRAGContextNumberPrompt,
    AnswerWithRAGContextBooleanPrompt,
    AnswerWithRAGContextNamesPrompt,
    AnswerWithRAGContextStringPrompt,
    RephrasedQuestionsPrompt
)
import re


class KnowledgeBase:
    """知识库主类"""
    
    def __init__(self):
        self.es_client = ElasticSearchClient()
        self.embedding_client = EmbeddingClient()
        self.qwen_client = QwenLLMClient()
        self.is_initialized = False
    
    def initialize(self):
        """初始化知识库"""
        try:
            # 创建向量索引，根据阿里text-embedding-v4的实际维度（1024维）
            self.es_client.create_index(dimension=1024)  # 阿里text-embedding-v4的维度是1024
            self.is_initialized = True
            logger.info("知识库初始化成功")
            return True
        except Exception as e:
            logger.error(f"知识库初始化失败: {str(e)}")
            return False
    
    def add_documents(self, documents: List[Any]):
        """添加文档到知识库"""
        if not self.is_initialized:
            if not self.initialize():
                raise RuntimeError("知识库未初始化")
        
        for i, doc in enumerate(documents):
            logger.info(f"正在处理文档 {i+1}/{len(documents)}")
            
            # 获取文档内容
            if hasattr(doc, 'page_content'):
                content = doc.page_content
            elif isinstance(doc, dict) and 'page_content' in doc:
                content = doc['page_content']
            elif isinstance(doc, str):
                content = doc
            else:
                content = str(doc)
            
            # 生成嵌入向量
            embedding = self.embedding_client.embed_query(content)
            
            # 获取元数据
            if hasattr(doc, 'metadata'):
                metadata = doc.metadata
            elif isinstance(doc, dict) and 'metadata' in doc:
                metadata = doc['metadata']
            else:
                metadata = {}
            
            # 添加到向量数据库
            self.es_client.add_document(
                content=content,
                vector=embedding,
                metadata=metadata
            )
        
        logger.info(f"成功添加 {len(documents)} 个文档到知识库")
    
    def load_and_add_documents(
        self, 
        source: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        从源文件/目录加载并添加文档
        
        Args:
            source: 源文件或目录路径
            chunk_size: 分块大小
            chunk_overlap: 分块重叠大小
        """
        if os.path.isdir(source):
            documents = DocumentLoader.load_documents_from_directory(
                source,
                use_advanced_splitting=True,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
            documents = DocumentLoader.load_document(
                source,
                use_advanced_splitting=True,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        
        self.add_documents(documents)
        logger.info(f"从 {source} 加载并添加文档完成，使用分块大小: {chunk_size}，重叠: {chunk_overlap}")
    
    def search(self, query: str, top_k: int = 5, use_reranker: bool = True, reranker_model: str = "default"):
        """搜索相关文档"""
        if not self.is_initialized:
            raise RuntimeError("知识库未初始化")
        
        # 生成查询向量
        query_vector = self.embedding_client.embed_query(query)
        
        # 执行混合搜索（可选择是否使用重排序）
        # 注意：reranker_model参数暂时未在es_client中使用，
        # 因为通用重排序器已在内部处理了模型选择
        results = self.es_client.hybrid_search(
            query_text=query,
            query_vector=query_vector,
            top_k=top_k,
            use_reranker=use_reranker
        )
        
        logger.info(f"搜索完成，返回 {len(results)} 个结果")
        return results
    
    def determine_answer_type(self, question: str) -> str:
        """
        根据问题内容判断答案类型
        
        Args:
            question: 问题文本
            
        Returns:
            答案类型 ("name", "number", "boolean", "names", "string")
        """
        question_lower = question.lower()
        
        # 数字类型关键词
        number_keywords = ["多少", "金额", "数值", "数量", "比例", "百分比", "率", "收入", "利润", "资产", "负债", 
                          "销售额", "成本", "费用", "投资", "市值", "股价", "收益", "产值", "产量", "销量"]
        
        # 名称类型关键词
        name_keywords = ["谁", "哪个", "哪位", "什么人", "姓名", "名字", "叫什么", "称谓", "职务", "职位", "角色"]
        
        # Boolean类型关键词
        boolean_keywords = ["是否", "有没有", "是否存在", "能否", "可否", "是不是", "有没有", "是否具备", "是否拥有"]
        
        # Names类型关键词
        names_keywords = ["哪些", "哪些人", "几个人", "都有谁", "都包括", "分别", "列表", "清单", "所有", "多个"]
        
        # 检查数字类型
        if any(keyword in question_lower for keyword in number_keywords):
            return "number"
        
        # 检查名称类型
        if any(keyword in question_lower for keyword in name_keywords):
            return "name"
        
        # 检查Boolean类型
        if any(keyword in question_lower for keyword in boolean_keywords):
            return "boolean"
        
        # 检查Names类型
        if any(keyword in question_lower for keyword in names_keywords):
            return "names"
        
        # 默认为字符串类型
        return "string"

    def ask(self, question: str, top_k: int = 5, use_reranker: bool = True, reranker_model: str = "default"):
        """提问并获取答案"""
        # 先搜索相关文档（根据参数决定是否使用重排序）
        search_results = self.search(question, top_k, use_reranker=use_reranker, reranker_model=reranker_model)
        
        # 构建上下文
        context_parts = []
        sources = []
        
        for result in search_results:
            context_parts.append(result['content'])
            # 优先使用重排序得分，如果没有则使用混合得分
            score = result.get('rerank_score', result.get('hybrid_score', 0.0))
            sources.append({
                'id': result['id'],
                'content': result['content'][:200] + "...",
                'score': score,
                'original_score': result.get('hybrid_score', 0.0),  # 保留原始得分
                'rerank_position': result.get('rerank_position'),   # 重排序位置
                'original_position': result.get('original_position'), # 原始位置
                'use_reranker': use_reranker  # 记录是否使用了重排序
            })
        
        context = "\n\n".join(context_parts)
        
        # 使用动态提示生成器根据文档内容生成适应性提示
        try:
            # 从prompts模块导入动态提示生成器
            from src.prompts import dynamic_prompt_generator
            
            # 分析上下文以生成适应性提示
            adaptive_prompt = dynamic_prompt_generator.generate_context_aware_prompt(
                question=question,
                context=context
            )
            
            # 使用适配的提示进行问答
            answer = self.qwen_client.chat_with_custom_prompt(adaptive_prompt)
            
            # 获取文档分析结果以提供更多上下文信息
            document_analysis = dynamic_prompt_generator.analyzer.analyze_document(context)
            
            return {
                'answer': answer,
                'sources': sources,
                'search_results': search_results,
                'document_analysis': document_analysis,  # 提供文档分析结果
                'prompt_strategy': 'adaptive',  # 标记使用了动态提示
                'answer_type': 'adaptive'  # 标记答案类型
            }
            
        except Exception as e:
            logger.error(f"动态提示问答失败: {str(e)}，回退到传统问答")
            
            # 回退到原有逻辑
            answer_type = self.determine_answer_type(question)
            
            try:
                # 使用结构化对话
                structured_response = self.qwen_client.structured_chat(
                    question=question, 
                    context=context, 
                    answer_type=answer_type
                )
                
                # 解析结构化响应
                parsed_response = self.qwen_client.parse_structured_response(
                    structured_response, 
                    answer_type=answer_type
                )
                
                # 获取最终答案
                final_answer = parsed_response.get('final_answer', '未能从上下文中找到相关信息。')
                
                return {
                    'answer': final_answer,
                    'sources': sources,
                    'search_results': search_results,
                    'structured_response': parsed_response,  # 保留结构化响应以供调试
                    'answer_type': answer_type  # 记录答案类型
                }
                
            except Exception as fallback_e:
                logger.error(f"结构化问答回退失败: {str(fallback_e)}，最终回退到基础问答")
                
                # 最终回退到基础问答
                answer = self.qwen_client.chat(question, context)
                
                return {
                    'answer': answer,
                    'sources': sources,
                    'search_results': search_results,
                    'answer_type': 'fallback'  # 标记为回退模式
                }