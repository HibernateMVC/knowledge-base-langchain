"""LangChain增强版知识库主类"""
from typing import List, Dict, Any
from .models.es_vector_store import ElasticSearchClient
from .utils.embedding_client import EmbeddingClient
from .utils.document_loader import DocumentLoader
from .utils.logger import logger
from .langchain_integration.qwen_model import QwenLLMWrapper
from .langchain_integration.es_vector_store_wrapper import ElasticSearchVectorStore
from .langchain_integration.chains import RAGChain, QuestionRephraseChain, ComparativeAnswerChain
import os
import sys


class LangChainEnhancedKnowledgeBase:
    """基于LangChain增强版的知识库主类"""
    
    def __init__(self):
        # 初始化原有组件
        self.es_client = ElasticSearchClient()
        self.embedding_client = EmbeddingClient()
        
        # 初始化LangChain组件
        self.llm = QwenLLMWrapper()
        self.vector_store = ElasticSearchVectorStore(
            es_client=self.es_client,
            embedding_function=None  # 使用内部的embedding_client
        )
        
        # 创建各种链
        self.rag_chain = RAGChain(
            llm=self.llm,
            vector_store=self.vector_store
        )
        
        self.question_rephrase_chain = QuestionRephraseChain(llm=self.llm)
        self.comparative_answer_chain = ComparativeAnswerChain(llm=self.llm)
        
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
        
        # 提取内容和元数据
        texts = []
        metadatas = []
        
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
            
            # 获取元数据
            if hasattr(doc, 'metadata'):
                metadata = doc.metadata
            elif isinstance(doc, dict) and 'metadata' in doc:
                metadata = doc['metadata']
            else:
                metadata = {}
            
            texts.append(content)
            metadatas.append(metadata)
        
        # 添加到向量存储
        ids = self.vector_store.add_texts(texts, metadatas)
        
        logger.info(f"成功添加 {len(documents)} 个文档到知识库")
        return ids
    
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
        if not self.is_initialized:
            if not self.initialize():
                raise RuntimeError("知识库未初始化")
        
        # 使用带有高级分割策略的DocumentLoader加载文档
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
        
        ids = self.add_documents(documents)
        logger.info(f"从 {source} 加载并添加文档完成，使用分块大小: {chunk_size}，重叠: {chunk_overlap}")
        return ids
    
    def search(self, query: str, top_k: int = 5, use_reranker: bool = True, reranker_model: str = "default"):
        """搜索相关文档"""
        if not self.is_initialized:
            raise RuntimeError("知识库未初始化")
        
        # 使用向量存储的混合搜索功能
        results = self.vector_store.hybrid_search(
            query_text=query,
            top_k=top_k,
            use_reranker=use_reranker,
            reranker_model=reranker_model
        )
        
        logger.info(f"搜索完成，返回 {len(results)} 个结果")
        return results
    
    def ask(self, question: str, top_k: int = 5, use_reranker: bool = True, reranker_model: str = "default"):
        """使用LangChain RAG链提问并获取答案"""
        # 优先使用动态提示生成器分析文档类型
        try:
            # 搜索相关文档
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
            
            # 使用动态提示生成器根据上下文生成适应性提示
            from .prompts import dynamic_prompt_generator
            adaptive_prompt = dynamic_prompt_generator.generate_context_aware_prompt(
                question=question,
                context=context
            )
            
            # 使用模型直接回答
            response = self.llm._call(adaptive_prompt)
            
            # 解析结构化响应
            from .langchain_integration.chains import StructuredOutputParser
            output_parser = StructuredOutputParser()
            structured_response = output_parser.parse(response)
            
            # 提取最终答案
            answer = structured_response.get('final_answer', '未能从上下文中找到相关信息。')
            
            return {
                'answer': answer,
                'sources': sources,
                'search_results': search_results,
                'structured_response': structured_response,
                'prompt_strategy': 'adaptive',  # 标记使用了动态提示
                'answer_type': 'adaptive'
            }
            
        except Exception as e:
            # 如果动态提示失败，回退到原有的RAG链逻辑
            # 使用LangChain RAG链来处理问题
            chain_input = {
                "question": question,
                "top_k": top_k,
                "use_reranker": use_reranker,
                "reranker_model": reranker_model
            }
            
            result = self.rag_chain(chain_input)
            
            return {
                'answer': result['answer'],
                'sources': result['sources'],
                'search_results': result['search_results'],
                'structured_response': result.get('structured_response', {}),
                'answer_type': result.get('answer_type', 'fallback')
            }
    
    def rephrase_question(self, question: str, companies: List[str]):
        """重写问题"""
        chain_input = {
            "question": question,
            "companies": companies
        }
        
        result = self.question_rephrase_chain(chain_input)
        return result['rephrased_questions']
    
    def get_comparative_answer(self, context: str, original_question: str):
        """获取比较类问题的最终答案"""
        chain_input = {
            "context": context,
            "question": original_question
        }
        
        result = self.comparative_answer_chain(chain_input)
        return result['final_answer']