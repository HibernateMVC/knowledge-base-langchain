# LLM客户端 - 通义千问
import dashscope
from langchain_core.prompts import PromptTemplate
from src.config.settings import Config
from src.utils.logger import logger
from src.prompts import (
    AnswerWithRAGContextNamePrompt,
    AnswerWithRAGContextNumberPrompt,
    AnswerWithRAGContextBooleanPrompt,
    AnswerWithRAGContextNamesPrompt,
    AnswerWithRAGContextStringPrompt,
    AnswerWithRAGContextSharedPrompt
)
import json
from typing import Dict, Any, Union


class QwenLLMClient:
    """通义千问语言模型客户端"""
    
    def __init__(self):
        # 设置API密钥
        dashscope.api_key = Config.DASHSCOPE_API_KEY
        self.model = Config.QWEN_MODEL_NAME
        
    def get_qa_chain(self, retriever):
        """获取问答链"""
        # 自定义提示模板
        template = """基于以下上下文信息回答问题:

{context}

问题: {question}
答案:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # 注意：这里我们需要使用兼容的LLM对象
        # 为了简化，直接使用chat方法
        logger.info("已准备问答链")
        return self
    
    def chat(self, question: str, context: str = ""):
        """直接对话"""
        try:
            if context:
                prompt = f"基于以下上下文信息回答问题:\n\n{context}\n\n问题: {question}\n答案:"
            else:
                prompt = f"问题: {question}\n答案:"
            
            logger.info(f"向模型发送请求: {question[:50]}...")
            
            # 使用dashscope直接调用
            response = dashscope.Generation.call(
                model=self.model,
                prompt=prompt,
                temperature=0.7,
                max_tokens=2000
            )
            
            if response.status_code == 200:
                response_text = response.output.text
                logger.info(f"收到模型响应: {response_text[:50]}...")
                return response_text
            else:
                error_msg = f"API调用失败: {response.code} - {response.message}"
                logger.error(error_msg)
                raise Exception(error_msg)
                
        except Exception as e:
            logger.error(f"对话请求失败: {str(e)}")
            raise
    
    def structured_chat(self, question: str, context: str = "", answer_type: str = "string"):
        """
        结构化对话，根据不同类型使用不同的提示
        
        Args:
            question: 问题
            context: 上下文
            answer_type: 答案类型 ("name", "number", "boolean", "names", "string")
        """
        try:
            # 根据问题类型选择适当的提示
            if answer_type == "name":
                prompt_class = AnswerWithRAGContextNamePrompt
            elif answer_type == "number":
                prompt_class = AnswerWithRAGContextNumberPrompt
            elif answer_type == "boolean":
                prompt_class = AnswerWithRAGContextBooleanPrompt
            elif answer_type == "names":
                prompt_class = AnswerWithRAGContextNamesPrompt
            else:  # 默认为 string
                prompt_class = AnswerWithRAGContextStringPrompt
            
            # 构建用户提示
            user_prompt = prompt_class.user_prompt.format(context=context, question=question)
            
            logger.info(f"向模型发送结构化请求: {question[:50]}...")
            
            # 使用dashscope直接调用
            response = dashscope.Generation.call(
                model=self.model,
                prompt=user_prompt,
                temperature=0.3,  # 更低的温度以获得更准确的结果
                max_tokens=2000
            )
            
            if response.status_code == 200:
                response_text = response.output.text
                logger.info(f"收到模型结构化响应: {response_text[:50]}...")
                return response_text
            else:
                error_msg = f"API调用失败: {response.code} - {response.message}"
                logger.error(error_msg)
                raise Exception(error_msg)
                
        except Exception as e:
            logger.error(f"结构化对话请求失败: {str(e)}")
            raise
    
    def chat_with_custom_prompt(self, custom_prompt: str):
        """
        使用自定义提示与模型对话
        
        Args:
            custom_prompt: 自定义提示
        """
        try:
            logger.info(f"向模型发送自定义提示请求: {custom_prompt[:50]}...")
            
            # 使用dashscope直接调用
            response = dashscope.Generation.call(
                model=self.model,
                prompt=custom_prompt,
                temperature=0.3,  # 降低温度以获得更准确的结构化输出
                max_tokens=2000
            )
            
            if response.status_code == 200:
                response_text = response.output.text
                logger.info(f"收到模型自定义响应: {response_text[:50]}...")
                return response_text
            else:
                error_msg = f"API调用失败: {response.code} - {response.message}"
                logger.error(error_msg)
                raise Exception(error_msg)
                
        except Exception as e:
            logger.error(f"自定义提示对话请求失败: {str(e)}")
            raise
    
    def parse_structured_response(self, response_text: str, answer_type: str = "string") -> Dict[str, Any]:
        """
        解析结构化的模型响应
        
        Args:
            response_text: 模型响应文本
            answer_type: 答案类型
        """
        try:
            # 尝试解析JSON响应
            # 有时模型可能在回答前后添加说明文字，我们需要提取JSON部分
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}")
            
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                json_str = response_text[start_idx:end_idx+1]
                parsed_response = json.loads(json_str)
                return parsed_response
            else:
                # 如果找不到JSON格式，尝试返回基本结构
                return {
                    "step_by_step_analysis": "无法解析响应格式",
                    "reasoning_summary": "响应格式异常",
                    "relevant_pages": [],
                    "final_answer": response_text
                }
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {str(e)}")
            logger.error(f"响应内容: {response_text}")
            return {
                "step_by_step_analysis": "JSON解析失败",
                "reasoning_summary": "无法解析模型响应",
                "relevant_pages": [],
                "final_answer": response_text
            }