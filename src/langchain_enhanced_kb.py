"""LangChain增强版知识库主类"""
from typing import List, Dict, Any
from .models.es_vector_store import ElasticSearchClient
from .utils.document_loader import DocumentLoader
from .utils.logger import logger
from .config.settings import Config
from .langchain_integration.qwen_model import QwenLLMWrapper
from .langchain_integration.embeddings import DashScopeEmbeddings
from .langchain_integration.es_vector_store_wrapper import ElasticSearchVectorStore
from .langchain_integration.chains import RAGChain, QuestionRephraseChain, ComparativeAnswerChain
import os
import sys


class LangChainEnhancedKnowledgeBase:
    """基于LangChain增强版的知识库主类"""
    
    def __init__(self):
        # 初始化原有组件
        self.es_client = ElasticSearchClient()
        
        # 初始化LangChain组件
        self.llm = QwenLLMWrapper()
        self.embeddings = DashScopeEmbeddings()
        self.vector_store = ElasticSearchVectorStore(
            es_client=self.es_client,
            embedding_function=self.embeddings
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
    
    def search(self, query: str, top_k: int = 3, use_reranker: bool = False, reranker_model: str = "default"):
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
    
    def ask(self, question: str, top_k: int = 3, use_reranker: bool = False, reranker_model: str = "default", conversation_history: List[Dict[str, str]] = None):
        """使用LangChain RAG链提问并获取答案"""
        if conversation_history is None:
            conversation_history = []
        
        # 优先使用动态提示生成器分析文档类型
        try:
            def _is_greeting(q: str) -> bool:
                qn = (q or "").strip().lower()
                if not qn:
                    return True
                # 极短寒暄/问候
                greetings = ["你好", "您好", "嗨", "hi", "hello", "hey", "在吗", "在么", "在不在"]
                return (len(qn) <= 6) and any(g in qn for g in greetings)

            def _general_chat_answer(user_q: str) -> Dict[str, Any]:
                # 构建包含历史对话的提示
                history_text = ""
                if conversation_history:
                    history_text = "对话历史：\n"
                    for msg in conversation_history[-4:]:  # 只保留最近4条消息
                        role = "用户" if msg["role"] == "user" else "助手"
                        history_text += f"{role}：{msg['content']}\n"
                    history_text += "\n"
                
                chat_prompt = (
                    "你是一个企业知识库助手。当前知识库中没有与用户问题直接相关的文档可以引用。\n"
                    "请作为一个友好、礼貌的智能助手，根据自己的常识和通用知识来回答用户的问题。\n"
                    "如果问题只是打招呼或寒暄，请用自然的中文简短回复即可。\n\n"
                    f"{history_text}"
                    f"用户问题：{user_q}\n"
                )
                answer_text = self.llm._call(chat_prompt)
                return {
                    "answer": answer_text,
                    "sources": [],
                    "search_results": [],
                    "structured_response": {},
                    "prompt_strategy": "general_chat",
                    "answer_type": "general_chat",
                }

            # 寒暄类问题直接走通用对话（避免无意义检索并返回 N/A）
            if _is_greeting(question):
                logger.info("检测到寒暄/问候类问题，切换为通用对话模式")
                return _general_chat_answer(question)

            # 搜索相关文档
            search_results = self.search(question, top_k, use_reranker=use_reranker, reranker_model=reranker_model)
            
            # 如果知识库中没有检索到相关文档，则切换为通用对话模式
            if not search_results:
                logger.info("未检索到相关文档，切换为通用对话模式回答用户问题")
                return _general_chat_answer(question)

            # 如果检索到了结果，但相关度非常低，也切换为通用对话模式
            try:
                best_score = max(
                    float(r.get("rerank_score", r.get("hybrid_score", 0.0)) or 0.0)
                    for r in search_results
                )
            except Exception:
                best_score = 0.0

            if best_score < Config.RAG_MIN_RELEVANCE_SCORE:
                logger.info(
                    f"检索结果相关度过低(best_score={best_score:.4f} < {Config.RAG_MIN_RELEVANCE_SCORE})，切换为通用对话模式"
                )
                return _general_chat_answer(question)
            
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
            
            # 构建包含历史对话的提示前缀
            history_text = ""
            if conversation_history:
                history_text = "对话历史：\n"
                for msg in conversation_history[-4:]:  # 只保留最近4条消息
                    role = "用户" if msg["role"] == "user" else "助手"
                    history_text += f"{role}：{msg['content']}\n"
                history_text += "\n"
            
            # 使用动态提示生成器根据上下文生成适应性提示
            from .prompts import dynamic_prompt_generator
            adaptive_prompt = dynamic_prompt_generator.generate_context_aware_prompt(
                question=question,
                context=context
            )
            
            # 在提示前面添加历史对话
            if history_text:
                adaptive_prompt = history_text + adaptive_prompt
            
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